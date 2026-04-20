"""
trainer.py - DreamBooth 학습 관리
학습 루프, 검증, 모니터링 기능
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import os
import gc
import json
import time
import random
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable, IO
import logging

from config import DreamBoothConfig
from model import MemoryOptimizedModel, ModelOptimizer, PipelineManager
from dataset import DreamBoothDataset, DatasetUtils


class DreamBoothTrainer:
    """DreamBooth 학습 관리 클래스"""

    def __init__(self, config: DreamBoothConfig, event_log_path: Optional[str] = None):
        self.config = config
        self.global_step = 0
        self.start_time = None

        # 이벤트 로그 (JSONL) — 서브프로세스→API 스트리밍용
        self.event_log_path: Optional[str] = event_log_path
        self._event_log_fp: Optional[IO] = None
        self._event_lock = threading.Lock()
        if self.event_log_path:
            Path(self.event_log_path).parent.mkdir(parents=True, exist_ok=True)
            # append 모드 + line buffering — 각 json.write 후 flush 보장
            self._event_log_fp = open(self.event_log_path, "a", buffering=1, encoding="utf-8")

        # 로깅 설정
        self.setup_logging()

        # 재현성 설정은 Accelerator 초기화 전에 seed 고정
        self.setup_reproducibility()

        # Accelerator 초기화
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs")
        )

        # 모델 및 컴포넌트
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.noise_scheduler = None

        # Weight dtype은 Accelerator 구성으로부터 결정
        self.weight_dtype = self._resolve_weight_dtype()

        # 캐시된 텍스트 임베딩 (CPU offload 모드)
        self.cached_instance_embeddings: Optional[torch.Tensor] = None
        self.cached_class_embeddings: Optional[torch.Tensor] = None

        # 검증 관련
        self.validation_pipeline = None

    def emit_event(self, event_type: str, **fields: Any) -> None:
        """JSONL 이벤트 한 줄을 event_log에 append. event_log_path 미설정 시 no-op.

        Main process에서만 기록 (분산/멀티GPU 상황 대비)."""
        if self._event_log_fp is None:
            return
        if not self.accelerator.is_main_process:
            return
        record = {
            "ts": time.time(),
            "type": event_type,
            "step": self.global_step,
            **fields,
        }
        try:
            with self._event_lock:
                self._event_log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._event_log_fp.flush()
        except Exception as e:  # 이벤트 로그 장애가 학습을 막으면 안 됨
            self.logger.warning(f"Failed to write event log: {e}")

    def close_event_log(self) -> None:
        if self._event_log_fp is not None:
            try:
                self._event_log_fp.close()
            except Exception:
                pass
            self._event_log_fp = None

    def _resolve_weight_dtype(self) -> torch.dtype:
        """Accelerator mixed_precision에 맞는 고정 가중치 dtype 반환"""
        mp = self.accelerator.mixed_precision if hasattr(self, 'accelerator') else self.config.mixed_precision
        if mp == "fp16":
            return torch.float16
        if mp == "bf16":
            return torch.bfloat16
        return torch.float32

    def setup_reproducibility(self) -> None:
        """시드 고정 및 (선택적) deterministic 모드"""
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if getattr(self.config, "deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as e:
                self.logger.warning(f"Deterministic algorithms not fully available: {e}")
        else:
            torch.backends.cudnn.benchmark = True
        
    def setup_logging(self) -> None:
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self) -> None:
        """모델 설정"""
        self.logger.info("Setting up model...")

        # Accelerator가 초기화된 지금 다시 한 번 weight_dtype 확정
        self.weight_dtype = self._resolve_weight_dtype()

        # 모델 로드 (VAE 포함 - 캐스팅/슬라이싱을 위해)
        self.model = MemoryOptimizedModel(self.config.pretrained_model_name_or_path)
        self.model.load_components(load_vae=True)

        # LoRA 주입 (메모리 최적화 이전에 수행 - gradient checkpointing과 상호작용)
        if self.config.use_lora:
            self.model.apply_lora(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )

        # 8GB VRAM 최적화 (precision 캐스팅 없이)
        ModelOptimizer.optimize_for_8gb_vram(self.model)

        # VAE 슬라이싱/타일링
        self.model.enable_vae_optimizations(
            slicing=self.config.enable_vae_slicing,
            tiling=self.config.enable_vae_tiling,
        )

        # 학습 준비 (freeze/train 모드 설정)
        self.model.prepare_for_training()

        # 동결 컴포넌트만 weight_dtype으로 캐스팅 (UNet은 FP32 유지 - LoRA 아닐 때)
        self.model.cast_frozen_components(self.weight_dtype)

        # Text encoder CPU offload - 고정 프롬프트를 미리 인코딩하고 GPU에서 해제
        if self.config.cpu_offload_text_encoder:
            self._precompute_text_embeddings()

        # 노이즈 스케줄러
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler"
        )

        self.logger.info(f"Model setup completed (weight_dtype={self.weight_dtype}, "
                         f"lora={self.config.use_lora})")

    def _precompute_text_embeddings(self) -> None:
        """고정 프롬프트의 텍스트 임베딩을 미리 계산하고 text_encoder를 CPU로 이동."""
        self.logger.info("Pre-computing text embeddings for CPU offload...")

        te = self.model.text_encoder.to(self.accelerator.device)
        tokenizer = self.model.tokenizer

        def encode(prompt: str) -> torch.Tensor:
            ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.accelerator.device)
            with torch.no_grad():
                return te(ids)[0].to(dtype=self.weight_dtype)

        self.cached_instance_embeddings = encode(self.config.instance_prompt)
        if self.config.with_prior_preservation:
            self.cached_class_embeddings = encode(self.config.class_prompt)

        # text_encoder를 CPU로 이동하여 VRAM 확보
        self.model.text_encoder = self.model.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self.logger.info("Text embeddings cached; text_encoder offloaded to CPU")
    
    def setup_optimizer_and_scheduler(self) -> None:
        """옵티마이저 및 스케줄러 설정"""
        self.logger.info("Setting up optimizer and scheduler...")

        trainable_params = self.model.get_trainable_parameters()

        # 옵티마이저
        if self.config.use_8bit_adam:
            self.optimizer = ModelOptimizer.get_8bit_adam_optimizer(
                trainable_params,
                lr=self.config.learning_rate
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08
            )
        
        # 스케줄러
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps
        )
        
        self.logger.info("Optimizer and scheduler setup completed")
    
    def setup_dataset(self) -> None:
        """데이터셋 설정"""
        self.logger.info("Setting up dataset...")
        
        # 데이터셋 생성
        train_dataset = DreamBoothDataset(
            instance_data_root=self.config.instance_data_dir,
            instance_prompt=self.config.instance_prompt,
            tokenizer=self.model.tokenizer,
            class_data_root=self.config.class_data_dir if self.config.with_prior_preservation else None,
            class_prompt=self.config.class_prompt if self.config.with_prior_preservation else None,
            size=self.config.resolution,
            center_crop=self.config.center_crop,
            train=True,
            with_prior_preservation=self.config.with_prior_preservation
        )
        
        # 데이터로더 생성
        self.train_dataloader = DatasetUtils.create_dataloader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        self.logger.info(f"Dataset setup completed. {len(train_dataset)} samples")
    
    def prepare_accelerator(self) -> None:
        """Accelerator 준비"""
        self.logger.info("Preparing accelerator...")
        
        # Accelerator로 준비
        self.model.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        
        self.logger.info("Accelerator preparation completed")
    
    def _encode_prompt_ids(self, prompt_ids: torch.Tensor, cached: Optional[torch.Tensor],
                           batch_size: int) -> torch.Tensor:
        """텍스트 임베딩 획득: 캐시된 값 우선, 없으면 text_encoder 호출."""
        if cached is not None:
            # cached: [1, seq, dim] → [B, seq, dim]
            if cached.dim() == 2:
                cached = cached.unsqueeze(0)
            return cached.expand(batch_size, -1, -1).to(dtype=self.weight_dtype)

        with torch.no_grad():
            ids = prompt_ids
            if ids.dim() == 3:
                ids = ids.squeeze(1)
            hidden = self.model.text_encoder(ids)[0]
            return hidden.to(dtype=self.weight_dtype)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """손실 계산.

        - VAE, text_encoder: weight_dtype (frozen)
        - UNet: Accelerator autocast가 내부에서 dtype 처리 (LoRA가 아니면 FP32 파라미터)
        - 최종 loss는 float32로 누적
        """
        device = self.accelerator.device
        weight_dtype = self.weight_dtype

        # 이미지 → latent
        with torch.no_grad():
            images = batch["instance_images"].to(device=device, dtype=weight_dtype)
            latents = self.model.vae.encode(images).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor

        # 노이즈
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 텍스트 임베딩 (캐시된 경우 재사용)
        encoder_hidden_states = self._encode_prompt_ids(
            batch["instance_prompt_ids"], self.cached_instance_embeddings, batch_size
        )

        # UNet 예측
        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 타겟
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Prior preservation
        if self.config.with_prior_preservation and "class_images" in batch:
            with torch.no_grad():
                class_images = batch["class_images"].to(device=device, dtype=weight_dtype)
                class_latents = self.model.vae.encode(class_images).latent_dist.sample()
                class_latents = class_latents * self.model.vae.config.scaling_factor

            class_noise = torch.randn_like(class_latents)
            class_timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (class_latents.shape[0],), device=device
            ).long()
            class_noisy_latents = self.noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

            class_encoder_hidden_states = self._encode_prompt_ids(
                batch["class_prompt_ids"], self.cached_class_embeddings, class_latents.shape[0]
            )
            class_model_pred = self.model.unet(
                class_noisy_latents, class_timesteps, class_encoder_hidden_states
            ).sample

            if self.noise_scheduler.config.prediction_type == "epsilon":
                class_target = class_noise
            else:
                class_target = self.noise_scheduler.get_velocity(class_latents, class_noise, class_timesteps)

            prior_loss = F.mse_loss(class_model_pred.float(), class_target.float(), reduction="mean")
            loss = loss + self.config.prior_loss_weight * prior_loss

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """학습 스텝"""
        with self.accelerator.accumulate(self.model.unet):
            loss = self.compute_loss(batch)
            self.accelerator.backward(loss)

            # Accelerator가 fp16 scaler의 unscale을 알아서 처리함 (mixed_precision 통일 후)
            if self.accelerator.sync_gradients:
                trainable = [p for p in self.model.unet.parameters() if p.requires_grad]
                self.accelerator.clip_grad_norm_(trainable, 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            return loss.detach().item()

    def validate(self) -> None:
        """검증 수행"""
        if self.global_step % self.config.validation_steps != 0:
            return

        self.logger.info("Running validation...")
        self.emit_event("validation_started")

        try:
            # 검증 이미지 생성
            images = self.generate_validation_images()
            self.emit_event("validation", images=images)

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.emit_event("validation_error", error=str(e))

    def generate_validation_images(self) -> list:
        """검증 이미지 생성. 저장된 파일 경로 리스트 반환."""
        saved: list = []
        # CPU offload 모드라면 text_encoder를 GPU로 임시 이동
        offloaded = self.config.cpu_offload_text_encoder
        if offloaded:
            self.model.text_encoder = self.model.text_encoder.to(self.accelerator.device)

        try:
            if self.validation_pipeline is None:
                pipeline_manager = PipelineManager(self.config.pretrained_model_name_or_path)
                self.validation_pipeline = pipeline_manager.load_pipeline()

            # 학습 중인 UNet과 VAE, text_encoder를 검증 파이프라인에 주입
            self.validation_pipeline.unet = self.accelerator.unwrap_model(self.model.unet)
            self.validation_pipeline.vae = self.model.vae
            self.validation_pipeline.text_encoder = self.model.text_encoder

            validation_prompts = [
                self.config.validation_prompt,
                f"{self.config.validation_prompt} wearing a suit",
                f"{self.config.validation_prompt} smiling",
            ]

            validation_dir = Path(self.config.output_dir) / "validation"
            validation_dir.mkdir(exist_ok=True)

            for i, prompt in enumerate(validation_prompts):
                try:
                    image = self.validation_pipeline(
                        prompt=prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=self.config.resolution,
                        width=self.config.resolution,
                        generator=torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed + i)
                    ).images[0]

                    out_path = validation_dir / f"step_{self.global_step:06d}_prompt_{i:02d}.png"
                    image.save(out_path)
                    saved.append(str(out_path))
                except Exception as e:
                    self.logger.error(f"Failed to generate validation image {i}: {e}")
        finally:
            if offloaded:
                self.model.text_encoder = self.model.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        return saved

    def save_checkpoint(self) -> None:
        """체크포인트 저장"""
        if self.global_step % self.config.checkpointing_steps != 0:
            return

        self.logger.info(f"Saving checkpoint at step {self.global_step}")

        # Accelerator 상태 저장
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)

        # 모델 체크포인트 저장
        self.model.save_checkpoint(self.config.output_dir, self.global_step)

        # 체크포인트 수 제한
        self.cleanup_old_checkpoints()

        self.emit_event("checkpoint", path=save_path)

    def cleanup_old_checkpoints(self) -> None:
        """오래된 체크포인트 정리"""
        if self.config.checkpoints_total_limit is None:
            return

        checkpoints_dir = Path(self.config.output_dir)
        checkpoints = [d for d in checkpoints_dir.iterdir() if d.name.startswith("checkpoint-")]

        if len(checkpoints) > self.config.checkpoints_total_limit:
            # 가장 오래된 체크포인트부터 삭제
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            for checkpoint in checkpoints[:-self.config.checkpoints_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                self.logger.info(f"Removed old checkpoint: {checkpoint}")

    def train(self) -> None:
        """메인 학습 함수"""
        self.logger.info("Starting DreamBooth training...")
        self.start_time = time.time()
        self.emit_event(
            "started",
            max_train_steps=self.config.max_train_steps,
            output_dir=self.config.output_dir,
            use_lora=self.config.use_lora,
            mixed_precision=self.config.mixed_precision,
            resolution=self.config.resolution,
            instance_prompt=self.config.instance_prompt,
        )

        try:
            # 설정
            self.setup_model()
            self.setup_optimizer_and_scheduler()
            self.setup_dataset()
            self.prepare_accelerator()

            # 메모리 사용량 로그
            memory_usage = ModelOptimizer.calculate_memory_usage()
            self.logger.info(f"Initial memory usage: {memory_usage}")

            # 학습 루프
            progress_bar = tqdm(
                range(self.config.max_train_steps),
                disable=not self.accelerator.is_local_main_process
            )

            running_loss = 0.0
            log_interval = 50

            for epoch in range(1000):  # 충분히 큰 에포크 수
                for step, batch in enumerate(self.train_dataloader):
                    # 학습 스텝
                    loss = self.training_step(batch)

                    # 동기화 체크
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        self.global_step += 1
                        running_loss += loss

                        # step 이벤트 (매 step — 너무 시끄러우면 샘플링 가능)
                        lr_now = self.lr_scheduler.get_last_lr()[0]
                        elapsed = time.time() - self.start_time
                        self.emit_event(
                            "step",
                            loss=float(loss),
                            lr=float(lr_now),
                            elapsed=elapsed,
                            max_steps=self.config.max_train_steps,
                        )

                        # 로그 출력
                        if self.global_step % log_interval == 0:
                            avg_loss = running_loss / log_interval
                            lr = self.lr_scheduler.get_last_lr()[0]
                            elapsed_time = time.time() - self.start_time

                            self.logger.info(
                                f"Step {self.global_step}/{self.config.max_train_steps}, "
                                f"Loss: {avg_loss:.4f}, LR: {lr:.2e}, "
                                f"Time: {elapsed_time:.1f}s"
                            )

                            # Tensorboard 로그
                            self.accelerator.log({
                                "train_loss": avg_loss,
                                "learning_rate": lr,
                                "step": self.global_step
                            }, step=self.global_step)

                            running_loss = 0.0

                        # 검증
                        self.validate()

                        # 체크포인트 저장
                        self.save_checkpoint()

                        # 메모리 정리
                        if self.global_step % 20 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()

                        # 학습 완료 체크
                        if self.global_step >= self.config.max_train_steps:
                            break

                if self.global_step >= self.config.max_train_steps:
                    break

            # 최종 모델 저장
            self.save_final_model()

            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {total_time:.1f} seconds")
            self.emit_event("completed", elapsed=total_time, output_dir=self.config.output_dir)

        except Exception as e:
            self.logger.exception("Training failed")
            self.emit_event("error", error=str(e), error_type=type(e).__name__)
            raise
        finally:
            self.close_event_log()

    def save_final_model(self) -> None:
        """최종 모델 저장.
        LoRA 모드: 어댑터 가중치만 저장 (base model 재사용).
        Full fine-tuning: 전체 파이프라인 저장."""
        self.logger.info("Saving final model...")
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        unwrapped_unet = self.accelerator.unwrap_model(self.model.unet)

        if self.config.use_lora:
            from peft.utils import get_peft_model_state_dict
            lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
            out_path = Path(self.config.output_dir) / "pytorch_lora_weights.bin"
            torch.save(lora_state_dict, out_path)
            self.logger.info(f"LoRA weights saved to {out_path}")
            return

        # Full fine-tuning: 전체 파이프라인 저장
        pipeline = self.model.create_pipeline(self.config.output_dir)
        pipeline.unet = unwrapped_unet
        pipeline.save_pretrained(self.config.output_dir)
        self.logger.info(f"Final model saved to {self.config.output_dir}")

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """체크포인트에서 재시작"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        # Accelerator 상태 로드
        self.accelerator.load_state(checkpoint_path)

        # 글로벌 스텝 복원
        step_from_path = int(Path(checkpoint_path).name.split("-")[1])
        self.global_step = step_from_path

        self.logger.info(f"Resumed from step {self.global_step}")


if __name__ == "__main__":
    # 학습 예제
    from config import PresetConfigs

    # 설정 로드
    config = PresetConfigs.get_person_config()
    config.max_train_steps = 400

    # 트레이너 생성 및 학습
    trainer = DreamBoothTrainer(config)
    trainer.train()
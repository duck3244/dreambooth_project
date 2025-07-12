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
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

from config import DreamBoothConfig
from model import MemoryOptimizedModel, ModelOptimizer, PipelineManager
from dataset import DreamBoothDataset, DatasetUtils


class DreamBoothTrainer:
    """DreamBooth 학습 관리 클래스"""
    
    def __init__(self, config: DreamBoothConfig):
        self.config = config
        self.global_step = 0
        self.start_time = None
        
        # 로깅 설정
        self.setup_logging()
        
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
        
        # 검증 관련
        self.validation_pipeline = None
        
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
        
        # 모델 로드
        self.model = MemoryOptimizedModel(self.config.pretrained_model_name_or_path)
        self.model.load_components(load_vae=False)  # VAE는 필요할 때만 로드
        
        # 8GB VRAM 최적화
        ModelOptimizer.optimize_for_8gb_vram(self.model)
        
        # 학습 준비
        self.model.prepare_for_training()
        
        # 노이즈 스케줄러
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler"
        )
        
        self.logger.info("Model setup completed")
    
    def setup_optimizer_and_scheduler(self) -> None:
        """옵티마이저 및 스케줄러 설정"""
        self.logger.info("Setting up optimizer and scheduler...")
        
        # 옵티마이저
        if self.config.use_8bit_adam:
            self.optimizer = ModelOptimizer.get_8bit_adam_optimizer(
                self.model.get_trainable_parameters(),
                lr=self.config.learning_rate
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.get_trainable_parameters(),
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
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """손실 계산"""
        # VAE 로드 (필요한 경우)
        if self.model.vae is None:
            from diffusers import AutoencoderKL
            self.model.vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="vae"
            )
            self.model.vae = self.model.vae.to(self.accelerator.device)
            self.model.vae.requires_grad_(False)

            # VAE도 동일한 precision으로 설정
            target_dtype = next(self.model.unet.parameters()).dtype
            self.model.vae = self.model.vae.to(dtype=target_dtype)

        # 모든 텐서를 동일한 dtype으로 통일
        target_dtype = next(self.model.unet.parameters()).dtype

        # 이미지를 latent space로 변환
        with torch.no_grad():
            # 이미지를 올바른 dtype으로 변환
            images = batch["instance_images"].to(dtype=target_dtype)
            latents = self.model.vae.encode(images).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor

        # 노이즈 추가
        noise = torch.randn_like(latents, dtype=target_dtype)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=latents.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 텍스트 임베딩
        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder(
                batch["instance_prompt_ids"].squeeze(1)
            )[0]
            # 텍스트 임베딩도 올바른 dtype으로 변환
            encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        # UNet 예측
        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 타겟 계산
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # 손실 계산 - 모든 텐서가 동일한 dtype인지 확인
        target = target.to(dtype=model_pred.dtype)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Prior preservation 손실 (사용하는 경우)
        if self.config.with_prior_preservation and "class_images" in batch:
            # 클래스 이미지에 대한 손실 계산 (유사한 방식)
            with torch.no_grad():
                class_images = batch["class_images"].to(dtype=target_dtype)
                class_latents = self.model.vae.encode(class_images).latent_dist.sample()
                class_latents = class_latents * self.model.vae.config.scaling_factor

            class_noise = torch.randn_like(class_latents, dtype=target_dtype)
            class_noisy_latents = self.noise_scheduler.add_noise(class_latents, class_noise, timesteps)

            with torch.no_grad():
                class_encoder_hidden_states = self.model.text_encoder(
                    batch["class_prompt_ids"].squeeze(1)
                )[0]
                class_encoder_hidden_states = class_encoder_hidden_states.to(dtype=target_dtype)

            class_model_pred = self.model.unet(class_noisy_latents, timesteps, class_encoder_hidden_states).sample

            if self.noise_scheduler.config.prediction_type == "epsilon":
                class_target = class_noise
            else:
                class_target = self.noise_scheduler.get_velocity(class_latents, class_noise, timesteps)

            class_target = class_target.to(dtype=class_model_pred.dtype)
            prior_loss = F.mse_loss(class_model_pred.float(), class_target.float(), reduction="mean")
            loss = loss + self.config.prior_loss_weight * prior_loss

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """학습 스텝"""
        with self.accelerator.accumulate(self.model.unet):
            # 손실 계산
            loss = self.compute_loss(batch)

            # 역전파
            self.accelerator.backward(loss)

            # 그래디언트 클리핑 (FP16 문제 해결)
            if self.accelerator.sync_gradients:
                try:
                    # 안전한 그래디언트 클리핑
                    if self.accelerator.scaler is not None:
                        # FP16 스케일러가 있는 경우 안전하게 처리
                        self.accelerator.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), 1.0)
                    else:
                        # 일반적인 클리핑
                        self.accelerator.clip_grad_norm_(self.model.unet.parameters(), 1.0)
                except RuntimeError as e:
                    if "Attempting to unscale FP16 gradients" in str(e):
                        # FP16 그래디언트 문제 시 스킵
                        print("Warning: Skipping gradient clipping due to FP16 issue")
                        pass
                    else:
                        raise e
                except ValueError as e:
                    if "Attempting to unscale FP16 gradients" in str(e):
                        # FP16 그래디언트 문제 시 스킵
                        print("Warning: Skipping gradient clipping due to FP16 issue")
                        pass
                    else:
                        raise e

            # 옵티마이저 스텝
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            return loss.detach().item()

    def validate(self) -> None:
        """검증 수행"""
        if self.global_step % self.config.validation_steps != 0:
            return

        self.logger.info("Running validation...")

        try:
            # 검증 이미지 생성
            self.generate_validation_images()

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")

    def generate_validation_images(self) -> None:
        """검증 이미지 생성"""
        if self.validation_pipeline is None:
            # 임시 파이프라인 생성
            pipeline_manager = PipelineManager(self.config.pretrained_model_name_or_path)
            self.validation_pipeline = pipeline_manager.load_pipeline()

            # 현재 학습된 UNet으로 교체
            self.validation_pipeline.unet = self.accelerator.unwrap_model(self.model.unet)

            # VAE 로드 (필요한 경우)
            if self.model.vae is None:
                from diffusers import AutoencoderKL
                self.model.vae = AutoencoderKL.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="vae"
                )
                self.model.vae = self.model.vae.to(self.accelerator.device)

            self.validation_pipeline.vae = self.model.vae

        # 검증 프롬프트
        validation_prompts = [
            self.config.validation_prompt,
            f"{self.config.validation_prompt} wearing a suit",
            f"{self.config.validation_prompt} smiling",
        ]

        # 이미지 생성
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
                    generator=torch.Generator().manual_seed(self.config.seed + i)
                ).images[0]

                image.save(validation_dir / f"step_{self.global_step:06d}_prompt_{i:02d}.png")

            except Exception as e:
                self.logger.error(f"Failed to generate validation image {i}: {e}")

        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

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

    def save_final_model(self) -> None:
        """최종 모델 저장"""
        self.logger.info("Saving final model...")

        # 파이프라인 생성 및 저장
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # VAE 로드 (저장을 위해)
            if self.model.vae is None:
                from diffusers import AutoencoderKL
                self.model.vae = AutoencoderKL.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="vae"
                )

            # 파이프라인 생성
            pipeline = self.model.create_pipeline(self.config.output_dir)
            pipeline.unet = self.accelerator.unwrap_model(self.model.unet)

            # 저장
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
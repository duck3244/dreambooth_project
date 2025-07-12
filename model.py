"""
model.py - DreamBooth 모델 관리
모델 로딩, 저장, 최적화 기능
"""

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Dict, Any, Tuple
import os
import gc
from pathlib import Path


class MemoryOptimizedModel:
    """메모리 최적화된 모델 관리 클래스"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 컴포넌트
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        
        # 메모리 최적화 플래그
        self.xformers_enabled = False
        self.gradient_checkpointing_enabled = False
        
    def load_components(self, load_vae: bool = True) -> None:
        """모델 컴포넌트 로드"""
        print("Loading model components...")
        
        # 토크나이저 로드
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            use_fast=False
        )
        
        # 텍스트 인코더 로드
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder"
        )
        
        # VAE 로드 (선택적)
        if load_vae:
            self.vae = AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae"
            )
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(self.device)
        
        # UNet 로드
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_name,
            subfolder="unet"
        )
        
        # 스케줄러 로드
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_name,
            subfolder="scheduler"
        )
        
        # 디바이스 이동
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        
        print("Model components loaded successfully!")
    
    def enable_memory_optimization(self) -> None:
        """메모리 최적화 활성화"""
        print("Enabling memory optimizations...")
        
        # Gradient checkpointing
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
            self.gradient_checkpointing_enabled = True
            print("✓ UNet gradient checkpointing enabled")
        
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
            print("✓ Text encoder gradient checkpointing enabled")
        
        # xFormers 최적화
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            self.xformers_enabled = True
            print("✓ xFormers memory efficient attention enabled")
        except Exception as e:
            print(f"✗ xFormers not available: {e}")
        
        # 혼합 정밀도 설정 - 일관성 있게 적용
        if self.device.type == 'cuda':
            # 모든 모델을 동일한 precision으로 설정
            try:
                self.unet = self.unet.to(dtype=torch.float16)
                self.text_encoder = self.text_encoder.to(dtype=torch.float16)
                if self.vae is not None:
                    self.vae = self.vae.to(dtype=torch.float16)
                print("✓ Mixed precision (fp16) enabled for all components")
            except Exception as e:
                print(f"✗ Mixed precision failed, using fp32: {e}")
                # FP32로 폴백
                self.unet = self.unet.to(dtype=torch.float32)
                self.text_encoder = self.text_encoder.to(dtype=torch.float32)
                if self.vae is not None:
                    self.vae = self.vae.to(dtype=torch.float32)
                print("✓ Using FP32 precision for stability")

        # CUDA 설정 최적화
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for CUDA operations")

    def prepare_for_training(self) -> None:
        """학습 준비"""
        # UNet만 학습 모드로 설정
        self.unet.train()
        self.unet.requires_grad_(True)

        # 다른 컴포넌트는 평가 모드
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        if self.vae is not None:
            self.vae.eval()
            self.vae.requires_grad_(False)

        print("Model prepared for training (UNet only)")

    def get_trainable_parameters(self) -> torch.nn.Parameter:
        """학습 가능한 파라미터 반환"""
        return self.unet.parameters()

    def save_checkpoint(self, output_dir: str, step: int) -> None:
        """체크포인트 저장"""
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # UNet 저장
        self.unet.save_pretrained(checkpoint_dir / "unet")

        # 메타데이터 저장
        metadata = {
            "step": step,
            "model_name": self.model_name,
            "xformers_enabled": self.xformers_enabled,
            "gradient_checkpointing_enabled": self.gradient_checkpointing_enabled
        }

        torch.save(metadata, checkpoint_dir / "metadata.pt")
        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """체크포인트 로드"""
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # UNet 로드
        self.unet = UNet2DConditionModel.from_pretrained(
            checkpoint_dir / "unet"
        )
        self.unet = self.unet.to(self.device)

        # 메타데이터 로드
        metadata_path = checkpoint_dir / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path)
            step = metadata.get("step", 0)
            print(f"Checkpoint loaded from step {step}")
            return step

        return 0

    def create_pipeline(self, output_dir: str) -> StableDiffusionPipeline:
        """학습된 모델로 파이프라인 생성"""
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            unet=self.unet,
            text_encoder=self.text_encoder,
            vae=self.vae,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )

        return pipeline

    def clear_memory(self) -> None:
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleared")


class ModelOptimizer:
    """모델 최적화 유틸리티"""

    @staticmethod
    def get_8bit_adam_optimizer(parameters, lr: float = 1e-6) -> torch.optim.Optimizer:
        """8-bit Adam 옵티마이저 생성"""
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08
            )
        except ImportError:
            print("Warning: bitsandbytes not available, using standard AdamW")
            return torch.optim.AdamW(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08
            )

    @staticmethod
    def calculate_memory_usage() -> Dict[str, float]:
        """GPU 메모리 사용량 계산"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_memory,
            "free_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        }

    @staticmethod
    def optimize_for_8gb_vram(model: MemoryOptimizedModel) -> None:
        """8GB VRAM 최적화 설정"""
        # 환경 변수 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        # 메모리 최적화 활성화
        model.enable_memory_optimization()

        # 추가 최적화
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # GPU 메모리의 95%만 사용

        print("8GB VRAM optimization applied")


class ModelValidator:
    """모델 검증 클래스"""

    @staticmethod
    def validate_model_components(model: MemoryOptimizedModel) -> Dict[str, bool]:
        """모델 컴포넌트 검증"""
        validation_result = {
            "tokenizer": model.tokenizer is not None,
            "text_encoder": model.text_encoder is not None,
            "vae": model.vae is not None,
            "unet": model.unet is not None,
            "scheduler": model.scheduler is not None
        }

        # 디바이스 확인
        if model.text_encoder is not None:
            validation_result["text_encoder_device"] = next(model.text_encoder.parameters()).device == model.device

        if model.unet is not None:
            validation_result["unet_device"] = next(model.unet.parameters()).device == model.device

        return validation_result

    @staticmethod
    def test_forward_pass(model: MemoryOptimizedModel, batch_size: int = 1) -> bool:
        """순전파 테스트"""
        try:
            # 더미 입력 생성
            dummy_text = "a photo of a person"
            dummy_image = torch.randn(batch_size, 3, 512, 512).to(model.device)

            # 텍스트 인코딩
            with torch.no_grad():
                text_inputs = model.tokenizer(
                    dummy_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(model.device)

                text_embeddings = model.text_encoder(text_inputs.input_ids)[0]

            # VAE 인코딩 (VAE가 로드된 경우)
            if model.vae is not None:
                with torch.no_grad():
                    latents = model.vae.encode(dummy_image.half()).latent_dist.sample()
                    latents = latents * model.vae.config.scaling_factor
            else:
                latents = torch.randn(batch_size, 4, 64, 64).to(model.device)

            # UNet 순전파
            timesteps = torch.randint(0, 1000, (batch_size,)).to(model.device)
            noise_pred = model.unet(latents, timesteps, text_embeddings).sample

            print("Forward pass test successful")
            return True

        except Exception as e:
            print(f"Forward pass test failed: {e}")
            return False


class PipelineManager:
    """파이프라인 관리 클래스"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None

    def load_pipeline(self) -> StableDiffusionPipeline:
        """파이프라인 로드"""
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )

            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")

            # 메모리 효율적 어텐션 활성화
            try:
                self.pipeline.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

            print(f"Pipeline loaded from {self.model_path}")
            return self.pipeline

        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None
    ) -> Optional[Any]:
        """이미지 생성"""
        if self.pipeline is None:
            print("Pipeline not loaded")
            return None

        try:
            # 시드 설정
            if seed is not None:
                torch.manual_seed(seed)

            # 이미지 생성
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )

            return result.images[0]

        except Exception as e:
            print(f"Image generation failed: {e}")
            return None

    def batch_generate(
        self,
        prompts: list,
        output_dir: str,
        **kwargs
    ) -> None:
        """배치 이미지 생성"""
        os.makedirs(output_dir, exist_ok=True)

        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")

            image = self.generate_image(prompt, **kwargs)
            if image is not None:
                image.save(os.path.join(output_dir, f"generated_{i:03d}.png"))

            # 메모리 정리
            if i % 5 == 0:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # 모델 테스트 예제
    print("Testing model components...")

    # 모델 생성 및 로드
    model = MemoryOptimizedModel()
    model.load_components(load_vae=False)  # 메모리 절약을 위해 VAE는 나중에 로드

    # 8GB VRAM 최적화
    ModelOptimizer.optimize_for_8gb_vram(model)

    # 학습 준비
    model.prepare_for_training()

    # 검증
    validation_result = ModelValidator.validate_model_components(model)
    print(f"Model validation: {validation_result}")

    # 메모리 사용량 확인
    memory_usage = ModelOptimizer.calculate_memory_usage()
    print(f"Memory usage: {memory_usage}")

    # 순전파 테스트
    forward_test = ModelValidator.test_forward_pass(model)
    print(f"Forward pass test: {'Passed' if forward_test else 'Failed'}")

    # 메모리 정리
    model.clear_memory()
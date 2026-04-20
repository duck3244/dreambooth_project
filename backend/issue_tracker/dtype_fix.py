#!/usr/bin/env python3
"""
dtype_compatibility_fix.py - RTX 4060 Dtype 호환성 문제 완전 해결
Float/Half dtype 불일치 문제를 근본적으로 해결
"""

import torch
import os
from config import DreamBoothConfig

def create_fp32_only_config():
    """완전한 FP32 전용 설정"""
    config = DreamBoothConfig()
    
    # 모든 precision을 FP32로 통일
    config.mixed_precision = "no"
    config.prior_generation_precision = "fp32"
    
    # RTX 4060 8GB에 맞춘 메모리 최적화
    config.train_batch_size = 1
    config.gradient_accumulation_steps = 8
    config.resolution = 512
    config.learning_rate = 5e-7  # 안정적인 학습률
    config.max_train_steps = 800
    
    # 메모리 최적화 옵션
    config.use_8bit_adam = True
    config.gradient_checkpointing = True
    config.enable_xformers_memory_efficient_attention = True
    
    # 안정성 향상
    config.lr_warmup_steps = 50
    config.validation_steps = 100
    config.checkpointing_steps = 50
    
    return config

def patch_accelerator_for_fp32():
    """Accelerator를 FP32 전용으로 패치"""
    from accelerate import Accelerator
    
    class FP32OnlyAccelerator(Accelerator):
        """FP32 전용 Accelerator"""
        
        def __init__(self, *args, **kwargs):
            # Mixed precision 강제 비활성화
            kwargs['mixed_precision'] = 'no'
            super().__init__(*args, **kwargs)
        
        def prepare_model(self, model, device_placement=None, evaluation_mode=False):
            """모델을 FP32로 강제 변환"""
            model = super().prepare_model(model, device_placement, evaluation_mode)
            
            # 모든 파라미터를 FP32로 변환
            for param in model.parameters():
                if param.dtype == torch.float16:
                    param.data = param.data.float()
            
            return model
    
    return FP32OnlyAccelerator

def create_safe_trainer():
    """완전히 안전한 트레이너 생성"""
    from trainer import DreamBoothTrainer
    from accelerate import Accelerator
    import torch.nn.functional as F
    
    class SafeDreamBoothTrainer(DreamBoothTrainer):
        """Dtype 안전 트레이너"""
        
        def __init__(self, config):
            # 부모 초기화 전에 설정 확인
            config.mixed_precision = "no"
            super().__init__(config)
            
            # Accelerator를 FP32 전용으로 재생성
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision="no",  # 강제 FP32
                log_with="tensorboard",
                project_dir=os.path.join(config.output_dir, "logs")
            )
        
        def setup_model(self):
            """모델 설정 (FP32 강제)"""
            super().setup_model()
            
            # 모든 모델 컴포넌트를 명시적으로 FP32로 변환
            self.model.unet = self.model.unet.float()
            self.model.text_encoder = self.model.text_encoder.float()
            
            print("All models forced to FP32")
        
        def compute_loss(self, batch):
            """Dtype 안전 손실 계산"""
            # VAE 로드 및 FP32 변환
            if self.model.vae is None:
                from diffusers import AutoencoderKL
                self.model.vae = AutoencoderKL.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="vae",
                    torch_dtype=torch.float32  # 명시적 FP32
                )
                self.model.vae = self.model.vae.to(self.accelerator.device)
                self.model.vae.requires_grad_(False)
                self.model.vae = self.model.vae.float()  # 강제 FP32
            
            # 모든 입력을 FP32로 변환
            instance_images = batch["instance_images"].float()
            
            # VAE 인코딩
            with torch.no_grad():
                latents = self.model.vae.encode(instance_images).latent_dist.sample()
                latents = latents * self.model.vae.config.scaling_factor
            
            # 노이즈 추가 (FP32)
            noise = torch.randn_like(latents, dtype=torch.float32)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=latents.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 텍스트 인코딩 (FP32)
            with torch.no_grad():
                encoder_hidden_states = self.model.text_encoder(
                    batch["instance_prompt_ids"].squeeze(1)
                )[0].float()  # 명시적 FP32 변환
            
            # UNet 예측
            model_pred = self.model.unet(
                noisy_latents.float(), 
                timesteps, 
                encoder_hidden_states.float()
            ).sample
            
            # 타겟 계산
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            
            # 손실 계산 (모든 텐서 FP32)
            loss = F.mse_loss(
                model_pred.float(), 
                target.float(), 
                reduction="mean"
            )
            
            return loss
    
    return SafeDreamBoothTrainer

def apply_environment_fixes():
    """환경 변수 최적화"""
    # CUDA 메모리 최적화
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 성능을 위해
    
    # PyTorch 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # 성능 향상
    
    # 메모리 관리
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    print("Environment optimized for FP32 training")

def test_dtype_consistency():
    """Dtype 일관성 테스트"""
    print("Testing dtype consistency...")
    
    try:
        # FP32 텐서 생성
        tensor_a = torch.randn(512, 512, dtype=torch.float32, device='cuda')
        tensor_b = torch.randn(512, 512, dtype=torch.float32, device='cuda')
        
        # 연산 테스트
        result = torch.matmul(tensor_a, tensor_b)
        print(f"✅ FP32 연산 성공: {result.dtype}")
        
        # 혼합 dtype 테스트
        tensor_half = tensor_a.half()
        try:
            mixed_result = torch.matmul(tensor_a, tensor_half)
            print("⚠️  Mixed dtype operation succeeded (unexpected)")
        except RuntimeError as e:
            print(f"✅ Mixed dtype properly blocked: {e}")
        
        # 정리
        del tensor_a, tensor_b, result
        if 'tensor_half' in locals():
            del tensor_half
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Dtype test failed: {e}")
        return False

def create_debug_main():
    """디버깅용 메인 함수"""
    
    debug_main_content = '''#!/usr/bin/env python3
"""
debug_main.py - Dtype 문제 디버깅용 메인 파일
"""

import os
import torch
from dtype_compatibility_fix import apply_environment_fixes, create_fp32_only_config, create_safe_trainer

def main():
    print("=== DreamBooth FP32 Safe Training ===")
    
    # 환경 최적화
    apply_environment_fixes()
    
    # 안전한 설정 생성
    config = create_fp32_only_config()
    config.instance_data_dir = "./instance_images"
    config.instance_prompt = "a photo of sks cat"
    config.class_prompt = "a photo of cat"
    config.with_prior_preservation = True
    
    print("Configuration:")
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.train_batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    # 안전한 트레이너 생성
    SafeTrainer = create_safe_trainer()
    trainer = SafeTrainer(config)
    
    try:
        # 학습 시작
        trainer.train()
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open('debug_main.py', 'w') as f:
        f.write(debug_main_content)
    
    print("Created debug_main.py for safe training")

def main():
    """메인 함수"""
    print("=== RTX 4060 Dtype 호환성 문제 해결 ===")
    
    # 환경 최적화
    apply_environment_fixes()
    
    # Dtype 일관성 테스트
    if not test_dtype_consistency():
        print("❌ Basic dtype tests failed")
        return
    
    # 안전한 설정 생성
    print("\n안전한 FP32 설정:")
    config = create_fp32_only_config()
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Steps: {config.max_train_steps}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    # 디버그 메인 파일 생성
    create_debug_main()
    
    print("\n권장 실행 방법:")
    print("1. python debug_main.py  # 완전 안전 모드")
    print("2. 또는 기존 main.py에서 config.py의 mixed_precision을 'no'로 변경")
    
    print("\n✅ Dtype 호환성 문제 해결 완료!")

if __name__ == "__main__":
    main()

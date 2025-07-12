#!/usr/bin/env python3
"""
gpu_compatibility_fix.py - RTX 4060 GPU 호환성 문제 해결
FP16 gradient scaling 문제 및 Ampere 아키텍처 최적화
"""

import torch
import os
from config import DreamBoothConfig, PresetConfigs

def create_rtx4060_optimized_config():
    """RTX 4060에 특화된 설정 생성"""
    config = PresetConfigs.get_object_config()
    
    # RTX 4060 최적화 설정
    config.mixed_precision = "no"  # FP16 문제 해결을 위해 비활성화
    config.gradient_accumulation_steps = 8  # 메모리 보상을 위해 증가
    config.learning_rate = 1e-6  # 더 안정적인 학습률
    config.max_train_steps = 600  # 스텝 수 증가로 보상
    config.train_batch_size = 1
    config.use_8bit_adam = True
    config.gradient_checkpointing = True
    config.enable_xformers_memory_efficient_attention = True
    
    # 추가 안정성 설정
    config.lr_warmup_steps = 100  # 워밍업 추가
    config.validation_steps = 100
    config.checkpointing_steps = 50  # 더 자주 저장
    
    return config

def create_alternative_fp16_config():
    """대안적 FP16 설정 (더 안전한 버전)"""
    config = PresetConfigs.get_object_config()
    
    # 안전한 FP16 설정
    config.mixed_precision = "fp16"
    config.prior_generation_precision = "fp32"  # VAE/텍스트 인코더는 FP32
    config.gradient_accumulation_steps = 6
    config.learning_rate = 5e-7  # 매우 낮은 학습률
    config.max_train_steps = 800
    
    return config

def apply_gpu_environment_fixes():
    """GPU 환경 변수 및 설정 최적화"""
    
    # CUDA 메모리 최적화
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 디버깅용
    
    # PyTorch 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 안정성을 위해 비활성화
    torch.backends.cudnn.deterministic = True
    
    # 메모리 사용량 제한
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        # 메모리 파편화 방지
        torch.cuda.empty_cache()
    
    print("RTX 4060 최적화 환경 설정 완료")

def check_gpu_compatibility():
    """GPU 호환성 체크"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability}")
    print(f"Total Memory: {total_memory:.1f}GB")
    
    issues = []
    
    # RTX 4060 특이사항 체크
    if "RTX 4060" in gpu_name:
        print("RTX 4060 detected - applying specific optimizations")
        if total_memory < 7.5:
            issues.append("Insufficient GPU memory")
    
    # Compute capability 체크
    if compute_capability[0] < 7:
        issues.append("GPU compute capability too low for optimal performance")
    
    return len(issues) == 0, issues

def create_safe_trainer_config():
    """안전한 트레이너 설정"""
    from trainer import DreamBoothTrainer
    
    class SafeDreamBoothTrainer(DreamBoothTrainer):
        """RTX 4060용 안전한 트레이너"""
        
        def __init__(self, config):
            super().__init__(config)
            # 더 안전한 Accelerator 설정
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision="no",  # FP16 비활성화
                log_with="tensorboard",
                project_dir=os.path.join(config.output_dir, "logs")
            )
        
        def training_step(self, batch):
            """더 안전한 학습 스텝"""
            with self.accelerator.accumulate(self.model.unet):
                # 손실 계산
                loss = self.compute_loss(batch)
                
                # NaN 체크
                if torch.isnan(loss):
                    self.logger.warning("NaN loss detected, skipping step")
                    return 0.0
                
                # 역전파
                self.accelerator.backward(loss)
                
                # 안전한 그래디언트 클리핑
                if self.accelerator.sync_gradients:
                    try:
                        # 그래디언트 노름 체크
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.unet.parameters(), 
                            max_norm=1.0
                        )
                        
                        if total_norm > 10.0:  # 비정상적으로 큰 그래디언트
                            self.logger.warning(f"Large gradient norm: {total_norm}")
                    
                    except Exception as e:
                        self.logger.warning(f"Gradient clipping failed: {e}")
                
                # 옵티마이저 스텝
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                return loss.detach().item()
    
    return SafeDreamBoothTrainer

def run_gpu_diagnostic():
    """GPU 진단 실행"""
    print("=== RTX 4060 GPU 진단 ===")
    
    # GPU 정보
    is_compatible, issues = check_gpu_compatibility()
    
    if not is_compatible:
        print("❌ GPU 호환성 문제:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # 메모리 테스트
    print("\n메모리 테스트...")
    try:
        # 작은 텐서로 테스트
        test_tensor = torch.randn(1024, 1024, device='cuda')
        test_result = torch.matmul(test_tensor, test_tensor)
        del test_tensor, test_result
        torch.cuda.empty_cache()
        print("✅ 기본 GPU 연산 테스트 통과")
    except Exception as e:
        print(f"❌ GPU 연산 테스트 실패: {e}")
        return False
    
    # FP16 테스트
    print("\nFP16 호환성 테스트...")
    try:
        test_tensor_fp16 = torch.randn(512, 512, device='cuda', dtype=torch.float16)
        test_result_fp16 = torch.matmul(test_tensor_fp16, test_tensor_fp16)
        del test_tensor_fp16, test_result_fp16
        torch.cuda.empty_cache()
        print("✅ FP16 연산 테스트 통과")
        fp16_support = True
    except Exception as e:
        print(f"⚠️  FP16 연산 문제 감지: {e}")
        print("   FP32 모드를 사용하는 것을 권장합니다.")
        fp16_support = False
    
    return fp16_support

def main():
    """메인 실행 함수"""
    print("RTX 4060 호환성 문제 해결 도구")
    print("=" * 40)
    
    # GPU 진단
    fp16_support = run_gpu_diagnostic()
    
    # 환경 최적화 적용
    apply_gpu_environment_fixes()
    
    # 권장 설정 생성
    if fp16_support:
        print("\n권장 설정: 안전한 FP16 모드")
        config = create_alternative_fp16_config()
    else:
        print("\n권장 설정: FP32 모드 (안전)")
        config = create_rtx4060_optimized_config()
    
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"Max Train Steps: {config.max_train_steps}")
    
    # 실행 명령어 제안
    print("\n권장 실행 명령어:")
    if fp16_support:
        print("python main.py --mode train --preset object_safe_fp16 \\")
    else:
        print("python main.py --mode train --preset object_fp32 \\")
    
    print('    --instance_prompt "a photo of sks cat" \\')
    print('    --class_prompt "a photo of cat" \\')
    print('    --with_prior_preservation')
    
    return config

if __name__ == "__main__":
    main()

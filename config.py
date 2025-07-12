"""
config.py - DreamBooth 학습 설정 파일
RTX 4060 8GB VRAM 최적화 설정
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DreamBoothConfig:
    """DreamBooth 학습 설정 클래스"""
    
    # 모델 설정
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    
    # 데이터 설정
    instance_data_dir: str = "./instance_images"
    class_data_dir: str = "./class_images"
    output_dir: str = "./dreambooth_output"
    
    # 프롬프트 설정
    instance_prompt: str = "a photo of sks person"
    class_prompt: str = "a photo of person"
    
    # 이미지 설정
    resolution: int = 512
    center_crop: bool = True
    
    # 학습 설정
    train_batch_size: int = 1
    sample_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # 메모리 보상을 위해 증가
    learning_rate: float = 1e-6  # 더 안정적인 학습률
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 400
    
    # 검증 설정
    validation_prompt: str = "a photo of sks person"
    num_validation_images: int = 2
    validation_steps: int = 100
    
    # 체크포인트 설정
    checkpointing_steps: int = 100
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None
    
    # 메모리 최적화 설정
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    mixed_precision: str = "no"  # "fp16" 대신 "no" 사용
    prior_generation_precision: str = "fp32"
    
    # Prior preservation 설정
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    
    # 기타 설정
    seed: int = 42
    local_rank: int = -1
    num_workers: int = 0
    logging_dir: str = "./logs"
    
    # 8GB VRAM 최적화 설정
    max_memory_mb: int = 7500  # 8GB 중 7.5GB 사용
    
    def __post_init__(self):
        """설정 후 처리"""
        # 디렉토리 생성
        os.makedirs(self.instance_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        if self.with_prior_preservation:
            os.makedirs(self.class_data_dir, exist_ok=True)
    
    def get_env_vars(self):
        """환경 변수 설정 반환"""
        return {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "TOKENIZERS_PARALLELISM": "false"
        }
    
    def to_dict(self):
        """설정을 딕셔너리로 변환"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }


# 사전 정의된 설정들
class PresetConfigs:
    """사전 정의된 설정 프리셋"""
    
    @staticmethod
    def get_person_config():
        """사람 학습용 설정"""
        config = DreamBoothConfig()
        config.instance_prompt = "a photo of sks person"
        config.class_prompt = "a photo of person"
        config.validation_prompt = "a photo of sks person"
        config.max_train_steps = 400
        return config
    
    @staticmethod
    def get_object_config():
        """객체 학습용 설정"""
        config = DreamBoothConfig()
        config.instance_prompt = "a photo of sks object"
        config.class_prompt = "a photo of object"
        config.validation_prompt = "a photo of sks object"
        config.max_train_steps = 600
        return config
    
    @staticmethod
    def get_style_config():
        """스타일 학습용 설정"""
        config = DreamBoothConfig()
        config.instance_prompt = "a painting in sks style"
        config.class_prompt = "a painting"
        config.validation_prompt = "a painting in sks style"
        config.max_train_steps = 800
        config.learning_rate = 1e-6
        return config
    
    @staticmethod
    def get_fast_config():
        """빠른 학습용 설정 (품질 다소 저하)"""
        config = DreamBoothConfig()
        config.resolution = 256
        config.max_train_steps = 200
        config.validation_steps = 50
        config.checkpointing_steps = 50
        return config
    
    @staticmethod
    def get_high_quality_config():
        """고품질 학습용 설정 (시간 더 소요)"""
        config = DreamBoothConfig()
        config.max_train_steps = 800
        config.learning_rate = 1e-6
        config.gradient_accumulation_steps = 8
        config.validation_steps = 200
        return config


# 설정 검증 함수
def validate_config(config: DreamBoothConfig) -> bool:
    """설정 유효성 검증"""
    errors = []
    
    # 필수 디렉토리 확인
    if not os.path.exists(config.instance_data_dir):
        errors.append(f"Instance data directory not found: {config.instance_data_dir}")
    
    # 인스턴스 이미지 확인
    if os.path.exists(config.instance_data_dir):
        image_files = [f for f in os.listdir(config.instance_data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if len(image_files) < 3:
            errors.append(f"Need at least 3 images, found {len(image_files)}")
    
    # 학습 파라미터 확인
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.max_train_steps <= 0:
        errors.append("Max train steps must be positive")
    
    if config.train_batch_size <= 0:
        errors.append("Train batch size must be positive")
    
    # 메모리 설정 확인
    if config.train_batch_size > 1 and config.resolution > 512:
        errors.append("For 8GB VRAM, use batch_size=1 with resolution=512")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # 설정 테스트
    config = PresetConfigs.get_person_config()
    print("Person training config:")
    print(config.to_dict())
    
    is_valid = validate_config(config)
    print(f"\nConfiguration valid: {is_valid}")

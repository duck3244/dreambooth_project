#!/usr/bin/env python3
"""
main.py - DreamBooth 메인 실행 파일
RTX 4060 8GB VRAM 최적화 버전
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# 프로젝트 모듈 import
from config import DreamBoothConfig, PresetConfigs, validate_config
from trainer import DreamBoothTrainer
from utils import ProjectSetup, ConfigValidator, SystemMonitor, ModelTester, ImageProcessor
from dataset import ImageValidator


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="DreamBooth Fine-tuning for RTX 4060 8GB")
    
    # 기본 설정
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "test", "setup", "validate"],
                       help="실행 모드")
    
    # 데이터 경로
    parser.add_argument("--instance_data_dir", type=str, default="./instance_images",
                       help="학습할 이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, default="./dreambooth_output",
                       help="출력 디렉토리")
    
    # 프롬프트
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person",
                       help="인스턴스 프롬프트")
    parser.add_argument("--class_prompt", type=str, default="a photo of person",
                       help="클래스 프롬프트")
    
    # 학습 설정
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                       help="학습률")
    parser.add_argument("--max_train_steps", type=int, default=400,
                       help="최대 학습 스텝")
    parser.add_argument("--resolution", type=int, default=512,
                       help="이미지 해상도")
    
    # 프리셋 설정
    parser.add_argument("--preset", type=str, default="person",
                       choices=["person", "object", "style", "fast", "high_quality"],
                       help="사전 정의된 설정 사용")
    
    # 기타 옵션
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="체크포인트에서 재시작")
    parser.add_argument("--validation_prompt", type=str, default=None,
                       help="검증용 프롬프트")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    parser.add_argument("--with_prior_preservation", action="store_true",
                       help="Prior preservation 사용")
    
    # 테스트 모드 옵션
    parser.add_argument("--test_model_path", type=str, default="./dreambooth_output",
                       help="테스트할 모델 경로")
    parser.add_argument("--test_prompts", type=str, nargs="+", 
                       default=["a photo of sks person"],
                       help="테스트용 프롬프트들")
    
    return parser.parse_args()


def setup_environment():
    """환경 설정"""
    # 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # PyTorch 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("Environment configured for 8GB VRAM optimization")


def create_config_from_args(args):
    """명령줄 인수에서 설정 생성"""
    # 프리셋 설정 로드
    if args.preset == "person":
        config = PresetConfigs.get_person_config()
    elif args.preset == "object":
        config = PresetConfigs.get_object_config()
    elif args.preset == "style":
        config = PresetConfigs.get_style_config()
    elif args.preset == "fast":
        config = PresetConfigs.get_fast_config()
    elif args.preset == "high_quality":
        config = PresetConfigs.get_high_quality_config()
    else:
        config = DreamBoothConfig()
    
    # 명령줄 인수로 오버라이드
    config.instance_data_dir = args.instance_data_dir
    config.output_dir = args.output_dir
    config.instance_prompt = args.instance_prompt
    config.class_prompt = args.class_prompt
    config.learning_rate = args.learning_rate
    config.max_train_steps = args.max_train_steps
    config.resolution = args.resolution
    config.seed = args.seed
    config.with_prior_preservation = args.with_prior_preservation
    config.resume_from_checkpoint = args.resume_from_checkpoint
    
    if args.validation_prompt:
        config.validation_prompt = args.validation_prompt
    
    return config


def setup_mode(args):
    """프로젝트 설정 모드"""
    print("=== DreamBooth Project Setup ===")
    
    # 프로젝트 구조 생성
    project_dir = ProjectSetup.create_project_structure("dreambooth_project")
    
    # 요구사항 체크
    print("\nChecking requirements...")
    requirements = ProjectSetup.check_requirements()
    all_good = True
    
    for req, status in requirements.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {req}: {status}")
        if not status:
            all_good = False
    
    if not all_good:
        print("\n❌ Some requirements are missing!")
        print("\nTo install required packages, run:")
        print(ProjectSetup.install_requirements())
        return False
    
    # 시스템 정보 출력
    print("\nSystem Information:")
    system_info = SystemMonitor.get_system_info()
    print(f"  CPU Usage: {system_info.get('cpu_percent', 'N/A')}%")
    print(f"  Memory: {system_info.get('memory_percent', 'N/A')}% used")
    if 'gpu_memory_percent' in system_info:
        print(f"  GPU Memory: {system_info['gpu_memory_percent']:.1f}% used")
        print(f"  GPU: {system_info.get('gpu_name', 'Unknown')}")
    
    print(f"\n✅ Setup completed! Project created at: {project_dir}")
    print("\nNext steps:")
    print("1. Copy your training images to instance_images/ folder")
    print("2. Run: python main.py --mode train")
    
    return True


def validate_mode(args):
    """검증 모드"""
    print("=== DreamBooth Validation ===")
    
    # 설정 생성
    config = create_config_from_args(args)
    
    # 설정 검증
    print("Validating configuration...")
    is_valid, issues = ConfigValidator.validate_training_setup(config)
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # 데이터셋 검증
    print("\nValidating dataset...")
    dataset_validation = ImageValidator.validate_dataset(config.instance_data_dir)
    
    if not dataset_validation["valid"]:
        print(f"❌ Dataset validation failed: {dataset_validation['error']}")
        return False
    
    print(f"✅ Dataset validation passed:")
    print(f"  - Total images: {dataset_validation['total_images']}")
    print(f"  - Valid images: {dataset_validation['valid_images']}")
    
    if dataset_validation["invalid_images"]:
        print(f"  - Invalid images: {dataset_validation['invalid_images']}")
        for img_path, error in dataset_validation["invalid_details"]:
            print(f"    * {img_path.name}: {error}")
    
    # 시스템 리소스 체크
    print("\nChecking system resources...")
    system_info = SystemMonitor.get_system_info()
    
    # GPU 메모리 체크
    if system_info.get("cuda_available", False):
        gpu_memory_gb = system_info.get("cuda_memory_reserved_mb", 0) / 1024
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {system_info.get('gpu_name', 'Unknown')}")
        print(f"  GPU Memory: {total_memory:.1f}GB total")
        
        if total_memory < 7:
            print("  ⚠️  Warning: GPU memory may be insufficient for training")
        else:
            print("  ✅ GPU memory sufficient")
    else:
        print("  ❌ CUDA not available")
        return False
    
    # RAM 체크
    memory_gb = system_info.get("memory_available_gb", 0)
    if memory_gb < 8:
        print(f"  ⚠️  Warning: Low RAM available: {memory_gb:.1f}GB")
    else:
        print(f"  ✅ RAM sufficient: {memory_gb:.1f}GB available")
    
    print("\n✅ All validations passed! Ready for training.")
    return True


def train_mode(args):
    """학습 모드"""
    print("=== DreamBooth Training ===")
    
    # 환경 설정
    setup_environment()
    
    # 설정 생성
    config = create_config_from_args(args)
    
    # 설정 검증
    print("Validating configuration...")
    if not validate_config(config):
        print("❌ Configuration validation failed!")
        return False
    
    # 학습 디렉토리 정보 출력
    print(f"\nTraining Configuration:")
    print(f"  Instance data: {config.instance_data_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Instance prompt: {config.instance_prompt}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_train_steps}")
    print(f"  Resolution: {config.resolution}")
    print(f"  Batch size: {config.train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    
    # 이미지 정보 출력
    image_files = [f for f in os.listdir(config.instance_data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"  Training images: {len(image_files)}")
    
    # 사용자 확인
    response = input("\nProceed with training? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return False
    
    # 학습 시작
    try:
        trainer = DreamBoothTrainer(config)
        
        # 체크포인트에서 재시작 (지정된 경우)
        if config.resume_from_checkpoint:
            trainer.resume_from_checkpoint(config.resume_from_checkpoint)
        
        # 학습 실행
        trainer.train()
        
        print("✅ Training completed successfully!")
        print(f"Model saved to: {config.output_dir}")
        
        # 테스트 이미지 생성
        print("\nGenerating test images...")
        test_model(config.output_dir, [config.validation_prompt])
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mode(args):
    """테스트 모드"""
    print("=== DreamBooth Testing ===")
    
    model_path = args.test_model_path
    test_prompts = args.test_prompts
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"Testing model: {model_path}")
    print(f"Test prompts: {test_prompts}")
    
    # 모델 테스터 생성
    tester = ModelTester(model_path)
    
    # 파이프라인 로드
    if not tester.load_pipeline():
        print("❌ Failed to load pipeline")
        return False
    
    # 출력 디렉토리 생성
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 생성 테스트
    print("Generating test images...")
    if not tester.test_generation(test_prompts, output_dir):
        print("❌ Image generation failed")
        return False
    
    # 성능 벤치마크
    print("Running performance benchmark...")
    benchmark_results = tester.benchmark_performance(test_prompts[0])
    
    print("\nBenchmark Results:")
    print(f"  Average time: {benchmark_results.get('avg_time_seconds', 0):.2f}s")
    print(f"  Memory usage: {benchmark_results.get('avg_memory_mb', 0):.1f}MB")
    
    print(f"\n✅ Testing completed! Results saved to: {output_dir}")
    return True


def test_model(model_path: str, prompts: list):
    """간단한 모델 테스트"""
    try:
        tester = ModelTester(model_path)
        if tester.load_pipeline():
            output_dir = os.path.join(os.path.dirname(model_path), "test_outputs")
            tester.test_generation(prompts, output_dir)
            print(f"Test images saved to: {output_dir}")
    except Exception as e:
        print(f"Test failed: {e}")


def process_images_if_needed(instance_data_dir: str):
    """필요한 경우 이미지 전처리"""
    # 이미지 파일 확인
    image_files = [f for f in os.listdir(instance_data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"❌ No images found in {instance_data_dir}")
        return False
    
    # 이미지 크기 체크
    need_processing = False
    for img_file in image_files:
        img_path = os.path.join(instance_data_dir, img_file)
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                if img.size[0] != 512 or img.size[1] != 512:
                    need_processing = True
                    break
        except Exception:
            need_processing = True
            break
    
    if need_processing:
        print("Some images need preprocessing...")
        response = input("Auto-resize images to 512x512? (y/N): ")
        
        if response.lower() == 'y':
            # 백업 디렉토리 생성
            backup_dir = instance_data_dir + "_backup"
            if not os.path.exists(backup_dir):
                import shutil
                shutil.copytree(instance_data_dir, backup_dir)
                print(f"Original images backed up to: {backup_dir}")
            
            # 이미지 처리
            processed_count = ImageProcessor.batch_process_images(
                instance_data_dir, instance_data_dir + "_processed", 512, True
            )
            
            if processed_count > 0:
                # 처리된 이미지로 교체
                import shutil
                shutil.rmtree(instance_data_dir)
                shutil.move(instance_data_dir + "_processed", instance_data_dir)
                print(f"✅ {processed_count} images processed and resized")
            
    return True


def main():
    """메인 함수"""
    print("🎨 DreamBooth Fine-tuning for RTX 4060 8GB")
    print("=" * 50)
    
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 모드별 실행
    success = False
    
    if args.mode == "setup":
        success = setup_mode(args)
    
    elif args.mode == "validate":
        success = validate_mode(args)
    
    elif args.mode == "train":
        # 이미지 전처리 확인
        if os.path.exists(args.instance_data_dir):
            if not process_images_if_needed(args.instance_data_dir):
                sys.exit(1)
        
        success = train_mode(args)
    
    elif args.mode == "test":
        success = test_mode(args)
    
    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)
    
    # 결과 출력
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n💥 Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
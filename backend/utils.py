"""
utils.py - DreamBooth 유틸리티 함수들
이미지 처리, 검증, 모니터링, 테스트 등의 유틸리티 기능
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import json
import time
import psutil
import GPUtil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import StableDiffusionPipeline


class ImageProcessor:
    """이미지 처리 유틸리티"""
    
    @staticmethod
    def resize_and_crop(image: Image.Image, target_size: int = 512) -> Image.Image:
        """이미지 크기 조정 및 중앙 크롭"""
        # 원본 비율 유지하면서 리사이즈
        img_ratio = image.width / image.height
        
        if img_ratio > 1:  # 가로가 더 긴 경우
            new_width = int(target_size * img_ratio)
            new_height = target_size
        else:  # 세로가 더 긴 경우
            new_width = target_size
            new_height = int(target_size / img_ratio)
        
        # 리사이즈
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 중앙 크롭
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def enhance_image(image: Image.Image, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
        """이미지 향상"""
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        return image
    
    @staticmethod
    def remove_background(image_path: str, output_path: str) -> bool:
        """배경 제거 (rembg 라이브러리 필요)"""
        try:
            from rembg import remove
            
            with open(image_path, 'rb') as input_file:
                input_data = input_file.read()
            
            output_data = remove(input_data)
            
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
            
            return True
        except ImportError:
            print("rembg library not installed. Use: pip install rembg")
            return False
        except Exception as e:
            print(f"Background removal failed: {e}")
            return False
    
    @staticmethod
    def batch_process_images(input_dir: str, output_dir: str, 
                           target_size: int = 512, enhance: bool = True) -> int:
        """이미지 배치 처리"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        processed_count = 0
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in supported_formats:
                try:
                    # 이미지 로드
                    image = Image.open(image_file)
                    
                    # RGB 변환
                    if image.mode != 'RGB':
                        if image.mode == 'RGBA':
                            # 흰색 배경으로 합성
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[-1])
                            image = background
                        else:
                            image = image.convert('RGB')
                    
                    # EXIF 정보 기반 회전
                    image = ImageOps.exif_transpose(image)
                    
                    # 크기 조정
                    image = ImageProcessor.resize_and_crop(image, target_size)
                    
                    # 향상 (선택적)
                    if enhance:
                        image = ImageProcessor.enhance_image(image, 
                                                           brightness=1.05, 
                                                           contrast=1.1, 
                                                           saturation=1.05)
                    
                    # 저장
                    output_file = output_path / f"processed_{image_file.stem}.jpg"
                    image.save(output_file, 'JPEG', quality=95)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Failed to process {image_file}: {e}")
        
        print(f"Processed {processed_count} images")
        return processed_count


class SystemMonitor:
    """시스템 모니터링 유틸리티"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 수집"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # GPU 정보 (가능한 경우)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_memory_total_mb": gpu.memoryTotal,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_utilization_percent": gpu.load * 100,
                    "gpu_temperature": gpu.temperature
                })
        except Exception:
            pass
        
        # PyTorch CUDA 정보
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2
            })
        else:
            info["cuda_available"] = False
        
        return info
    
    @staticmethod
    def monitor_training(log_file: str = "system_monitor.log", interval: int = 60):
        """학습 중 시스템 모니터링"""
        while True:
            info = SystemMonitor.get_system_info()
            
            # 로그 파일에 기록
            with open(log_file, 'a') as f:
                f.write(json.dumps(info) + '\n')
            
            # 메모리 사용량이 90% 이상이면 경고
            if info["memory_percent"] > 90:
                print(f"⚠️  High memory usage: {info['memory_percent']:.1f}%")
            
            if "gpu_memory_percent" in info and info["gpu_memory_percent"] > 90:
                print(f"⚠️  High GPU memory usage: {info['gpu_memory_percent']:.1f}%")
            
            time.sleep(interval)
    
    @staticmethod
    def plot_training_metrics(log_file: str, output_dir: str = "./plots"):
        """학습 메트릭 시각화"""
        try:
            # 로그 파일 읽기
            data = []
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except:
                        continue
            
            if not data:
                print("No data found in log file")
                return
            
            # 데이터 추출
            timestamps = [datetime.fromisoformat(d["timestamp"]) for d in data]
            cpu_usage = [d.get("cpu_percent", 0) for d in data]
            memory_usage = [d.get("memory_percent", 0) for d in data]
            gpu_memory = [d.get("gpu_memory_percent", 0) for d in data]
            
            # 플롯 생성
            os.makedirs(output_dir, exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # CPU 사용률
            axes[0, 0].plot(timestamps, cpu_usage, 'b-', linewidth=2)
            axes[0, 0].set_title('CPU Usage (%)')
            axes[0, 0].set_ylabel('Percentage')
            axes[0, 0].grid(True)
            
            # 메모리 사용률
            axes[0, 1].plot(timestamps, memory_usage, 'r-', linewidth=2)
            axes[0, 1].set_title('RAM Usage (%)')
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].grid(True)
            
            # GPU 메모리 사용률
            if any(gpu_memory):
                axes[1, 0].plot(timestamps, gpu_memory, 'g-', linewidth=2)
                axes[1, 0].set_title('GPU Memory Usage (%)')
                axes[1, 0].set_ylabel('Percentage')
                axes[1, 0].grid(True)
            
            # 전체 시스템 상태
            axes[1, 1].plot(timestamps, cpu_usage, 'b-', label='CPU', alpha=0.7)
            axes[1, 1].plot(timestamps, memory_usage, 'r-', label='RAM', alpha=0.7)
            if any(gpu_memory):
                axes[1, 1].plot(timestamps, gpu_memory, 'g-', label='GPU Memory', alpha=0.7)
            axes[1, 1].set_title('System Overview')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'system_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Metrics plot saved to {output_dir}/system_metrics.png")
            
        except Exception as e:
            print(f"Failed to plot metrics: {e}")


class ModelTester:
    """모델 테스트 유틸리티"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
    
    def load_pipeline(self) -> bool:
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
            
            # 메모리 효율성 최적화
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            print(f"Pipeline loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            return False
    
    def test_generation(self, prompts: List[str], output_dir: str = "./test_outputs") -> bool:
        """이미지 생성 테스트"""
        if self.pipeline is None:
            print("Pipeline not loaded")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for i, prompt in enumerate(prompts):
                print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
                
                # 이미지 생성
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(42 + i)
                ).images[0]
                
                # 저장
                filename = f"test_{i:03d}_{prompt[:50].replace(' ', '_')}.png"
                image.save(os.path.join(output_dir, filename))
                
                # 메모리 정리
                if i % 3 == 0:
                    torch.cuda.empty_cache()
            
            print(f"Test images saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Image generation test failed: {e}")
            return False
    
    def benchmark_performance(self, prompt: str, num_runs: int = 5) -> Dict[str, float]:
        """성능 벤치마크"""
        if self.pipeline is None:
            print("Pipeline not loaded")
            return {}
        
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            start_time = time.time()
            
            # 이미지 생성
            _ = self.pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            "avg_time_seconds": np.mean(times),
            "std_time_seconds": np.std(times),
            "avg_memory_mb": np.mean(memory_usage),
            "std_memory_mb": np.std(memory_usage)
        }


class ProjectSetup:
    """프로젝트 설정 유틸리티"""
    
    @staticmethod
    def create_project_structure(project_name: str = "dreambooth_project") -> str:
        """프로젝트 디렉토리 구조 생성"""
        base_dir = Path(project_name)
        
        directories = [
            "instance_images",
            "class_images", 
            "outputs",
            "checkpoints",
            "validation",
            "logs",
            "test_results",
            "scripts"
        ]
        
        for directory in directories:
            (base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        # README 파일 생성
        readme_content = f"""# {project_name}

DreamBooth Fine-tuning Project

## Directory Structure
- `instance_images/`: 학습할 이미지들 (3-5장 권장)
- `class_images/`: Prior preservation용 클래스 이미지들
- `outputs/`: 학습된 모델 출력
- `checkpoints/`: 중간 체크포인트들
- `validation/`: 검증 이미지들
- `logs/`: 학습 로그들
- `test_results/`: 테스트 결과 이미지들
- `scripts/`: 실행 스크립트들

## Usage
1. `instance_images/` 폴더에 학습할 이미지 넣기
2. `python main.py` 실행
3. 결과는 `outputs/` 폴더에서 확인

## Requirements
- NVIDIA GPU with 8GB+ VRAM
- Python 3.9+
- PyTorch with CUDA support
"""
        
        with open(base_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print(f"Project structure created at: {base_dir.absolute()}")
        return str(base_dir.absolute())
    
    @staticmethod
    def check_requirements() -> Dict[str, bool]:
        """요구사항 체크"""
        requirements = {
            "python_version": False,
            "torch_available": False,
            "cuda_available": False,
            "diffusers_available": False,
            "accelerate_available": False,
            "transformers_available": False
        }
        
        # Python 버전 체크 (3.8+)
        import sys
        if sys.version_info >= (3, 8):
            requirements["python_version"] = True
        
        # 라이브러리 체크
        try:
            import torch
            requirements["torch_available"] = True
            requirements["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import diffusers
            requirements["diffusers_available"] = True
        except ImportError:
            pass
        
        try:
            import accelerate
            requirements["accelerate_available"] = True
        except ImportError:
            pass
        
        try:
            import transformers
            requirements["transformers_available"] = True
        except ImportError:
            pass
        
        return requirements
    
    @staticmethod
    def install_requirements() -> str:
        """설치 명령어 생성"""
        install_commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install diffusers[training]",
            "pip install transformers accelerate",
            "pip install bitsandbytes xformers",
            "pip install matplotlib seaborn pillow",
            "pip install psutil gputil"
        ]
        
        return "\n".join(install_commands)


class ConfigValidator:
    """설정 검증 유틸리티"""
    
    @staticmethod
    def validate_training_setup(config) -> Tuple[bool, List[str]]:
        """학습 설정 검증"""
        issues = []
        
        # 디렉토리 체크
        if not os.path.exists(config.instance_data_dir):
            issues.append(f"Instance data directory not found: {config.instance_data_dir}")
        else:
            # 이미지 파일 수 체크
            image_files = [f for f in os.listdir(config.instance_data_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if len(image_files) < 3:
                issues.append(f"Need at least 3 images, found {len(image_files)}")
        
        # GPU 메모리 체크
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory_gb < 7:
                issues.append(f"GPU memory too low: {total_memory_gb:.1f}GB (need 8GB+)")
        else:
            issues.append("CUDA not available")
        
        # 학습 파라미터 체크
        if config.train_batch_size > 1 and config.resolution > 512:
            issues.append("For 8GB VRAM, use batch_size=1 with resolution=512")
        
        if config.learning_rate > 1e-4:
            issues.append(f"Learning rate too high: {config.learning_rate} (recommend < 1e-4)")
        
        # 출력 디렉토리 권한 체크
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            test_file = os.path.join(config.output_dir, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            issues.append(f"Cannot write to output directory: {e}")
        
        return len(issues) == 0, issues


if __name__ == "__main__":
    # 유틸리티 테스트
    print("=== DreamBooth Utils Test ===")
    
    # 요구사항 체크
    print("\n1. Checking requirements...")
    requirements = ProjectSetup.check_requirements()
    for req, status in requirements.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {req}: {status}")
    
    # 시스템 정보
    print("\n2. System information...")
    system_info = SystemMonitor.get_system_info()
    print(f"  CPU Usage: {system_info.get('cpu_percent', 'N/A')}%")
    print(f"  Memory Usage: {system_info.get('memory_percent', 'N/A')}%")
    if 'gpu_memory_percent' in system_info:
        print(f"  GPU Memory: {system_info['gpu_memory_percent']:.1f}%")
    
    # 프로젝트 구조 생성 예제
    print("\n3. Creating project structure...")
    project_dir = ProjectSetup.create_project_structure("test_project")
    print(f"  Project created at: {project_dir}")
    
    print("\nUtils test completed!")
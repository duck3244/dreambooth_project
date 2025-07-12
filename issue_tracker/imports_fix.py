#!/usr/bin/env python3
"""
import_fixes.py - Import 문제 수정 및 누락된 라이브러리 설치 확인
"""

import sys
import subprocess
import importlib

def check_and_install_packages():
    """필요한 패키지들 확인 및 설치 안내"""
    required_packages = {
        'torch': 'torch>=2.0.0',
        'diffusers': 'diffusers[training]>=0.21.0',
        'transformers': 'transformers>=4.25.0',
        'accelerate': 'accelerate>=0.20.0',
        'PIL': 'Pillow>=9.0.0',
        'numpy': 'numpy>=1.21.0',
        'psutil': 'psutil>=5.9.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0',
        'tqdm': 'tqdm>=4.64.0'
    }
    
    optional_packages = {
        'bitsandbytes': 'bitsandbytes>=0.35.0',
        'xformers': 'xformers>=0.0.20',
        'GPUtil': 'GPUtil>=1.4.0'
    }
    
    missing_required = []
    missing_optional = []
    
    print("Checking required packages...")
    for package, version in required_packages.items():
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_required.append(version)
    
    print("\nChecking optional packages...")
    for package, version in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (optional)")
            missing_optional.append(version)
    
    if missing_required:
        print("\n❌ Missing required packages:")
        for pkg in missing_required:
            print(f"  pip install {pkg}")
        return False
    
    if missing_optional:
        print("\n⚠️  Missing optional packages (recommended for 8GB VRAM optimization):")
        for pkg in missing_optional:
            print(f"  pip install {pkg}")
    
    return True

def fix_import_issues():
    """Import 문제들을 수정하는 함수"""
    fixes = []
    
    # utils.py 수정
    utils_fixes = """
# utils.py 파일 상단에 다음 import 추가:
from typing import List, Tuple, Dict, Optional, Union, Any

# GPUtil이 없는 경우를 위한 안전한 import:
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring disabled.")

# matplotlib 백엔드 설정 (GUI 없는 환경용):
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서도 작동
import matplotlib.pyplot as plt
"""
    fixes.append(("utils.py", utils_fixes))
    
    # config.py 수정
    config_fixes = """
# config.py 파일에서 dataclasses import 확인:
from dataclasses import dataclass
from typing import Optional
"""
    fixes.append(("config.py", config_fixes))
    
    # model.py 수정
    model_fixes = """
# model.py 파일에서 bitsandbytes 안전한 import:
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Using standard optimizers.")
"""
    fixes.append(("model.py", model_fixes))
    
    return fixes

def create_safe_utils():
    """안전한 utils.py 버전 생성"""
    safe_utils_content = '''"""
utils.py - DreamBooth 유틸리티 함수들 (Import 안전 버전)
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
from datetime import datetime

# 안전한 import들
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    class GPUtil:
        @staticmethod
        def getGPUs():
            return []

try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 환경
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. Model testing disabled.")


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
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[-1])
                            image = background
                        else:
                            image = image.convert('RGB')
                    
                    # EXIF 정보 기반 회전
                    image = ImageOps.exif_transpose(image)
                    
                    # 크기 조정
                    image = ImageProcessor.resize_and_crop(image, target_size)
                    
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
        if GPU_AVAILABLE:
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


class ModelTester:
    """모델 테스트 유틸리티"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
    
    def load_pipeline(self) -> bool:
        """파이프라인 로드"""
        if not DIFFUSERS_AVAILABLE:
            print("Diffusers not available for model testing")
            return False
            
        try:
            from diffusers import StableDiffusionPipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
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
                
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    generator=torch.Generator().manual_seed(42 + i)
                ).images[0]
                
                filename = f"test_{i:03d}_{prompt[:50].replace(' ', '_')}.png"
                image.save(os.path.join(output_dir, filename))
                
                if i % 3 == 0:
                    torch.cuda.empty_cache()
            
            print(f"Test images saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Image generation test failed: {e}")
            return False


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
        
        return len(issues) == 0, issues


if __name__ == "__main__":
    print("=== Import Fixes and Requirements Check ===")
    
    # 패키지 체크
    if check_and_install_packages():
        print("✅ All required packages are available")
    else:
        print("❌ Some packages are missing")
    
    # 수정 사항 출력
    fixes = fix_import_issues()
    print("\\nSuggested fixes:")
    for filename, fix in fixes:
        print(f"\\n{filename}:")
        print(fix)
'''
    
    return safe_utils_content

if __name__ == "__main__":
    print("=== DreamBooth Import Issues Fix ===")
    
    # 패키지 확인
    if not check_and_install_packages():
        print("\n❌ Please install missing packages first!")
        sys.exit(1)
    
    # 수정 사항 출력
    print("\n=== Suggested Import Fixes ===")
    fixes = fix_import_issues()
    for filename, fix in fixes:
        print(f"\n📝 {filename}:")
        print(fix)
    
    # 안전한 utils.py 생성 옵션
    response = input("\nCreate safe utils.py? (y/N): ")
    if response.lower() == 'y':
        safe_content = create_safe_utils()
        with open('utils_safe.py', 'w') as f:
            f.write(safe_content)
        print("✅ Created utils_safe.py with safe imports")
    
    print("\n✅ Import fixes completed!")

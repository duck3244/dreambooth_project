"""
dataset.py - DreamBooth 데이터셋 처리
이미지 로딩, 전처리, 데이터셋 생성 기능
"""

import os
import torch
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional, Tuple
import random

class DreamBoothDataset(Dataset):
    """DreamBooth 학습용 데이터셋"""
    
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = True,
        train: bool = True,
        with_prior_preservation: bool = False
    ):
        self.instance_data_root = Path(instance_data_root)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.train = train
        self.with_prior_preservation = with_prior_preservation
        
        # 인스턴스 이미지 경로 수집
        self.instance_images_path = self._collect_images(self.instance_data_root)
        self.num_instance_images = len(self.instance_images_path)
        
        print(f"Found {self.num_instance_images} instance images")
        
        # Prior preservation 설정
        if self.with_prior_preservation:
            self.class_data_root = Path(class_data_root) if class_data_root else None
            self.class_prompt = class_prompt
            if self.class_data_root:
                self.class_images_path = self._collect_images(self.class_data_root)
                self.num_class_images = len(self.class_images_path)
                print(f"Found {self.num_class_images} class images")
            else:
                self.class_images_path = []
                self.num_class_images = 0
        else:
            self.class_images_path = []
            self.num_class_images = 0
        
        # 이미지 전처리 파이프라인
        self.image_transforms = self._create_transforms()
        
        # 데이터셋 크기 계산
        self._length = max(self.num_instance_images, self.num_class_images)
    
    def _collect_images(self, data_root: Path) -> List[Path]:
        """디렉토리에서 이미지 파일 수집"""
        if not data_root.exists():
            return []
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = []
        
        for file_path in data_root.iterdir():
            if file_path.suffix.lower() in supported_formats:
                images.append(file_path)
        
        images.sort()  # 일관된 순서 보장
        return images
    
    def _create_transforms(self) -> transforms.Compose:
        """이미지 전처리 파이프라인 생성"""
        transform_list = []
        
        # 크기 조정
        if self.center_crop:
            transform_list.extend([
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size)
            ])
        else:
            transform_list.append(
                transforms.Resize(
                    (self.size, self.size),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        
        # 학습 시 데이터 증강
        if self.train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ])
        
        # 텐서 변환 및 정규화
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def _load_and_preprocess_image(self, image_path: Path) -> torch.Tensor:
        """이미지 로드 및 전처리"""
        try:
            image = Image.open(image_path)
            
            # RGBA를 RGB로 변환
            if image.mode == 'RGBA':
                # 흰색 배경으로 합성
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # EXIF 정보 기반 회전 처리
            image = ImageOps.exif_transpose(image)
            
            return self.image_transforms(image)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 기본 이미지 생성 (검은색)
            return torch.zeros(3, self.size, self.size)
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """프롬프트 토큰화"""
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> dict:
        example = {}
        
        # 인스턴스 이미지 처리
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = self._load_and_preprocess_image(instance_image_path)
        
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self._tokenize_prompt(self.instance_prompt)
        
        # Prior preservation 처리
        if self.with_prior_preservation and self.num_class_images > 0:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = self._load_and_preprocess_image(class_image_path)
            
            example["class_images"] = class_image
            example["class_prompt_ids"] = self._tokenize_prompt(self.class_prompt)
        
        return example


class ImageValidator:
    """이미지 유효성 검증 클래스"""
    
    @staticmethod
    def validate_image(image_path: Path) -> Tuple[bool, str]:
        """개별 이미지 유효성 검증"""
        if not image_path.exists():
            return False, "File does not exist"
        
        try:
            with Image.open(image_path) as img:
                # 기본 검증
                if img.size[0] < 256 or img.size[1] < 256:
                    return False, f"Image too small: {img.size}"
                
                # 형식 검증
                if img.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                    return False, f"Unsupported format: {img.format}"
                
                # 모드 검증
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    return False, f"Unsupported mode: {img.mode}"
                
                return True, "Valid"
                
        except Exception as e:
            return False, f"Error opening image: {e}"
    
    @staticmethod
    def validate_dataset(data_root: str) -> dict:
        """데이터셋 전체 유효성 검증"""
        data_path = Path(data_root)
        
        if not data_path.exists():
            return {"valid": False, "error": "Directory does not exist"}
        
        # 이미지 파일 수집
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in data_path.iterdir()
            if f.suffix.lower() in supported_formats
        ]
        
        if len(image_files) < 3:
            return {
                "valid": False,
                "error": f"Need at least 3 images, found {len(image_files)}"
            }
        
        # 개별 이미지 검증
        valid_images = []
        invalid_images = []
        
        for image_file in image_files:
            is_valid, message = ImageValidator.validate_image(image_file)
            if is_valid:
                valid_images.append(image_file)
            else:
                invalid_images.append((image_file, message))
        
        return {
            "valid": len(valid_images) >= 3,
            "total_images": len(image_files),
            "valid_images": len(valid_images),
            "invalid_images": len(invalid_images),
            "invalid_details": invalid_images
        }


class DatasetUtils:
    """데이터셋 유틸리티 함수들"""
    
    @staticmethod
    def create_dataloader(dataset: DreamBoothDataset, batch_size: int = 1, 
                         shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
        """데이터로더 생성"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    @staticmethod
    def preview_dataset(dataset: DreamBoothDataset, num_samples: int = 3):
        """데이터셋 미리보기"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        
        for i in range(num_samples):
            sample = dataset[i]
            image = sample["instance_images"]
            
            # 텐서를 이미지로 변환
            image = (image + 1) / 2  # [-1, 1] -> [0, 1]
            image = image.permute(1, 2, 0)  # CHW -> HWC
            
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"Sample {i+1}")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_dataset(data_root: str) -> dict:
        """데이터셋 분석"""
        data_path = Path(data_root)
        
        if not data_path.exists():
            return {"error": "Directory does not exist"}
        
        # 이미지 파일 수집
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in data_path.iterdir()
            if f.suffix.lower() in supported_formats
        ]
        
        if not image_files:
            return {"error": "No image files found"}
        
        # 이미지 크기 분석
        sizes = []
        formats = {}
        modes = {}
        
        for image_file in image_files:
            try:
                with Image.open(image_file) as img:
                    sizes.append(img.size)
                    formats[img.format] = formats.get(img.format, 0) + 1
                    modes[img.mode] = modes.get(img.mode, 0) + 1
            except Exception:
                continue
        
        # 통계 계산
        if sizes:
            widths, heights = zip(*sizes)
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            min_width, max_width = min(widths), max(widths)
            min_height, max_height = min(heights), max(heights)
        else:
            avg_width = avg_height = 0
            min_width = max_width = min_height = max_height = 0
        
        return {
            "total_images": len(image_files),
            "valid_images": len(sizes),
            "average_size": (avg_width, avg_height),
            "size_range": {
                "width": (min_width, max_width),
                "height": (min_height, max_height)
            },
            "formats": formats,
            "modes": modes
        }


if __name__ == "__main__":
    # 데이터셋 검증 예제
    data_root = "./instance_images"
    
    print("Validating dataset...")
    validation_result = ImageValidator.validate_dataset(data_root)
    print(f"Validation result: {validation_result}")
    
    if validation_result["valid"]:
        print("\nAnalyzing dataset...")
        analysis = DatasetUtils.analyze_dataset(data_root)
        print(f"Analysis result: {analysis}")

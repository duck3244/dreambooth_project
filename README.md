# DreamBooth Fine-tuning for RTX 4060 8GB

RTX 4060 8GB VRAM에 최적화된 DreamBooth fine-tuning 프로젝트입니다.

## 🎯 주요 특징

- **8GB VRAM 최적화**: 메모리 효율적인 설정으로 RTX 4060에서 안정적 실행
- **모듈화된 구조**: 기능별로 분리된 파일 구조로 유지보수 용이
- **자동 검증**: 데이터셋과 설정 자동 검증 기능
- **실시간 모니터링**: 학습 중 시스템 리소스 모니터링
- **사전 정의 프리셋**: 용도별 최적화된 설정 제공

## 📁 파일 구조

```
dreambooth_project/
├── config.py          # 설정 관리
├── dataset.py         # 데이터셋 처리
├── model.py           # 모델 관리
├── trainer.py         # 학습 관리
├── utils.py           # 유틸리티 함수들
├── main.py            # 메인 실행 파일
├── requirements.txt   # 필요 패키지
├── README.md          # 이 파일
├── instance_images/   # 학습할 이미지들
├── outputs/          # 학습된 모델 출력
├── checkpoints/      # 중간 체크포인트
├── validation/       # 검증 이미지들
├── logs/            # 학습 로그들
└── test_results/    # 테스트 결과
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
conda create -n dreambooth python=3.9
conda activate dreambooth

# 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers[training] transformers accelerate
pip install bitsandbytes xformers
pip install matplotlib seaborn pillow psutil GPUtil
```

### 2. 프로젝트 설정

```bash
# 프로젝트 구조 생성 및 요구사항 확인
python main.py --mode setup
```

### 3. 학습 이미지 준비

- `instance_images/` 폴더에 3-5장의 이미지 넣기
- 권장: 512x512 해상도, 다양한 각도와 조명
- 지원 형식: JPG, PNG, BMP, TIFF

### 4. 학습 실행

```bash
# 기본 사람 학습
python main.py --mode train --preset person

# 커스텀 설정
python main.py --mode train \
    --instance_prompt "a photo of sks person" \
    --learning_rate 2e-6 \
    --max_train_steps 400
```

### 5. 모델 테스트

```bash
# 학습된 모델 테스트
python main.py --mode test \
    --test_model_path ./dreambooth_output \
    --test_prompts "a photo of sks person" "sks person wearing a suit"
```

## ⚙️ 설정 옵션

### 사전 정의 프리셋

| 프리셋 | 용도 | 특징 |
|--------|------|------|
| `person` | 사람 학습 | 기본 설정, 400 스텝 |
| `object` | 객체 학습 | 600 스텝, 객체 최적화 |
| `style` | 스타일 학습 | 800 스텝, 낮은 학습률 |
| `fast` | 빠른 학습 | 200 스텝, 256 해상도 |
| `high_quality` | 고품질 | 800 스텝, 높은 accumulation |

### 주요 파라미터

```python
# 메모리 최적화 설정
train_batch_size = 1              # 8GB VRAM용
gradient_accumulation_steps = 4   # 효과적 배치 크기
use_8bit_adam = True             # 메모리 절약
gradient_checkpointing = True     # 메모리 효율성
mixed_precision = "fp16"         # 16비트 연산

# 학습 설정
learning_rate = 2e-6             # 안정적 학습률
max_train_steps = 400            # 과적합 방지
resolution = 512                 # 표준 해상도
```

## 📊 모니터링

### 학습 진행 상황

- **Tensorboard**: `./outputs/logs`에서 학습 메트릭 확인
- **검증 이미지**: `./validation`에 주기적으로 생성
- **체크포인트**: `./checkpoints`에 중간 저장

### 시스템 리소스

```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# 메모리 사용량 확인
python -c "from utils import SystemMonitor; print(SystemMonitor.get_system_info())"
```

## 🔧 문제 해결

### 메모리 부족 오류

```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 배치 크기 줄이기
python main.py --mode train --train_batch_size 1 --gradient_accumulation_steps 8
```

### 품질 개선

1. **이미지 품질 향상**
   - 고해상도 원본 이미지 사용
   - 다양한 각도와 조명 조건
   - 배경이 단순한 이미지 선호

2. **학습 파라미터 조정**
   - 학습률 낮추기: `--learning_rate 1e-6`
   - 스텝 수 증가: `--max_train_steps 600`
   - Prior preservation 사용: `--with_prior_preservation`

### 일반적인 오류

| 오류 | 해결방법 |
|------|----------|
| CUDA out of memory | 배치 크기 줄이기, gradient accumulation 증가 |
| xformers not available | `pip install xformers` 재설치 |
| 이미지 로드 실패 | 이미지 형식 확인, 손상된 파일 제거 |
| 학습 불안정 | 학습률 낮추기, warmup steps 추가 |

## 📝 사용 예제

### 1. 인물 학습

```bash
# 3-5장의 인물 사진을 instance_images/에 넣고
python main.py --mode train \
    --preset person \
    --instance_prompt "a photo of sks john" \
    --max_train_steps 400
```

### 2. 애완동물 학습

```bash
python main.py --mode train \
    --preset object \
    --instance_prompt "a photo of sks dog" \
    --class_prompt "a photo of dog" \
    --with_prior_preservation
```

### 3. 예술 스타일 학습

```bash
python main.py --mode train \
    --preset style \
    --instance_prompt "a painting in sks style" \
    --learning_rate 1e-6 \
    --max_train_steps 800
```

## 🎨 결과 활용

### 이미지 생성

```python
from diffusers import StableDiffusionPipeline
import torch

# 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "./dreambooth_output",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 이미지 생성
image = pipe(
    "a photo of sks person wearing a red shirt",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated.png")
```

### 배치 생성

```bash
python main.py --mode test \
    --test_model_path ./dreambooth_output \
    --test_prompts \
        "a photo of sks person" \
        "sks person in a suit" \
        "portrait of sks person" \
        "sks person smiling"
```

## 📈 성능 최적화

### RTX 4060 8GB 최적 설정

```python
# config.py에서
train_batch_size = 1
gradient_accumulation_steps = 4
resolution = 512
use_8bit_adam = True
gradient_checkpointing = True
enable_xformers_memory_efficient_attention = True
mixed_precision = "fp16"
```

### 학습 시간 단축

- **해상도 낮추기**: 256x256 사용
- **스텝 수 줄이기**: 200-300 스텝
- **빠른 프리셋 사용**: `--preset fast`

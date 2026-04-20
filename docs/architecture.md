# Architecture

현재 레포는 "8GB VRAM에서 동작하는 DreamBooth 학습/추론 엔진"과 그것을
브라우저에서 조작할 수 있는 "웹 애플리케이션"을 하나의 모노레포로 묶은
구조입니다.

- **학습/추론 코어** (`backend/config.py`, `dataset.py`, `model.py`, `trainer.py`,
  `utils.py`, `main.py`) — CLI로도 독립 실행 가능.
- **API 레이어** (`backend/api/**`) — FastAPI. **torch를 import하지 않고**
  코어 모듈을 subprocess로 띄워 GPU 소유권을 격리.
- **프런트엔드** (`frontend/**`) — Vite + React 18 + TypeScript + Tailwind.
  React Router로 페이지 전환, Query로 폴링, EventSource로 SSE 구독.

---

## 1. 레이어 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (Vite dev: :5173)                    │
│  React Pages  Dashboard / Datasets / TrainNew / TrainLive /     │
│               Models / Generate                                 │
│  api.ts (fetch) ──►  /api/*  (Vite proxy → :8765)               │
│  useSSE.ts  (EventSource) ──►  /api/train|inference/{id}/events │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  HTTP / SSE
┌─────────────────────────────────────────────────────────────────┐
│        FastAPI (uvicorn :8765)  —  no torch import              │
│  routes/  health  gpu  datasets  train  models  inference       │
│  concurrency.require_gpu_free  →  409 if any worker alive       │
│  JobManager            InferenceManager          (stateful)     │
│     │ spawns                   │ spawns                         │
│     ▼                          ▼                                │
│  subprocess                 subprocess                          │
│  backend.api.workers        backend.api.workers                 │
│  .train_worker              .infer_worker                       │
│     │  imports core                                             │
│     ▼                                                           │
│  config · dataset · model · trainer · utils                     │
│     │  writes JSONL                                             │
│     ▼                                                           │
│  <JOBS_DIR>/<id>/events.jsonl   <INFERENCE_DIR>/<id>/events.jsonl│
│     ▲  tail_jsonl() → SSE                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  GPU only in worker subprocess
┌─────────────────────────────────────────────────────────────────┐
│  NVIDIA GPU (RTX 4060 8GB)                                      │
│  diffusers Stable Diffusion pipeline / DreamBooth UNet          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 디렉토리 상세

### `backend/` — 학습/추론 코어 (CLI 호환)

| 파일 | 책임 |
|------|------|
| `config.py` | `DreamBoothConfig` (dataclass) + `PresetConfigs` (person/object/style/fast/high_quality) + `validate_config()` |
| `dataset.py` | `DreamBoothDataset` (torch Dataset), `ImageValidator`, `DatasetUtils` |
| `model.py` | `MemoryOptimizedModel` (UNet/VAE/TextEncoder 로딩, LoRA/xformers/VAE slicing 등), `ModelOptimizer`, `ModelValidator`, `PipelineManager` |
| `trainer.py` | `DreamBoothTrainer` — 학습 루프, validation, checkpointing, **event log emit** |
| `utils.py` | `ImageProcessor`, `SystemMonitor`, `ModelTester`, `ProjectSetup`, `ConfigValidator` |
| `main.py` | argparse CLI (train / test / setup / validate) |

### `backend/api/` — HTTP/SSE 레이어 (torch 미사용)

| 파일 | 책임 |
|------|------|
| `app.py` | FastAPI `create_app()` + `lifespan()` (`JobManager` / `InferenceManager` 초기화, reconcile, 백그라운드 reaper) |
| `settings.py` | 경로·환경 변수 (`DREAMBOOTH_*`), `ensure_dirs()` |
| `gpu.py` | pynvml 기반 GPU 상태 조회 |
| `paths.py` | `safe_component()`, `safe_filename()`, `resolve_under()` — path traversal 방지 |
| `state.py` | `read_json()`, `write_json_atomic()` (tempfile → fsync → `os.replace`) |
| `event_log.py` | `sse_frame()`, `tail_jsonl()` — JSONL을 비동기로 tailing |
| `concurrency.py` | `require_gpu_free()` → 409 Conflict |
| `job_manager.py` | 학습 subprocess 수명 주기 (`start` / `is_alive` / `stop` / `reap_finished` / `reconcile_on_startup`) |
| `inference_manager.py` | 추론 subprocess 수명 주기 + 생성 이미지 리스트 집계 |
| `schemas.py` | Pydantic 요청/응답 모델 |
| `routes/` | `health`, `gpu`, `datasets`, `train`, `models`, `inference` |
| `workers/train_worker.py` | `python -m` 엔트리: config JSON 로드 → `DreamBoothTrainer.train()` |
| `workers/infer_worker.py` | `python -m` 엔트리: request JSON 로드 → diffusers 파이프라인 실행 |

### `frontend/src/`

| 경로 | 내용 |
|------|------|
| `pages/Dashboard.tsx` | GPU 상태 카드, 최근 job 요약 |
| `pages/Datasets.tsx` | 데이터셋 생성 / 이미지 업로드 / 삭제 |
| `pages/TrainNew.tsx` | 학습 파라미터 폼 → `POST /api/train` |
| `pages/TrainLive.tsx` | SSE 구독, step/loss 차트(recharts), 이벤트 로그 |
| `pages/Models.tsx` | 완료된 모델(LoRA / full) 목록 |
| `pages/Generate.tsx` | 프롬프트 입력 → SSE 이미지 스트리밍 |
| `components/` | `GpuStatusCard`, `JobStatusBadge`, `LossChart` |
| `hooks/useSSE.ts` | EventSource 래퍼 — type별 콜백 / 자동 close |
| `api.ts`, `types.ts` | `/api/*` 클라이언트 + 타입 |

### `scripts/`

- `dev.sh` — backend + frontend를 하나의 터미널에서 실행, Ctrl+C로 둘 다 종료
- `dev_backend.sh` — conda env (`py310_pt`) activate 후 `uvicorn --reload`
- `dev_frontend.sh` — `npm install` 후 `vite`

---

## 3. 핵심 설계 결정

### 3.1 GPU 격리: API는 torch를 import하지 않는다

FastAPI 프로세스가 torch를 import하면
1. 기동이 느려지고,
2. uvicorn `--reload`가 워커를 재시작할 때마다 수 초씩 블록되고,
3. 무엇보다 API 프로세스가 CUDA 컨텍스트를 점유해 학습/추론 워커가
   OOM을 맞는다.

따라서 **모든 torch 사용은 `backend.api.workers.*`의 자식 프로세스에서**
일어납니다. API는 `subprocess.Popen(..., start_new_session=True)`으로
새 프로세스 그룹을 띄우고, `SIGTERM`을 리더에게 보내면 자식 트리까지
깔끔히 회수됩니다.

### 3.2 단일 GPU, 단일 작업

8GB VRAM에서는 학습과 추론이 동시에 돌면 둘 다 OOM에 빠집니다.
`concurrency.require_gpu_free()`가 `JobManager.has_running_job()` /
`InferenceManager.has_running_job()`를 확인해, 이미 작업 중이면
**HTTP 409 Conflict**로 거절합니다.

### 3.3 JSONL 이벤트 로그 → SSE

- 학습 워커는 진행 상황(`start`, `step`, `validation`, `checkpoint`,
  `completed`, `error`, `cancelled`)을 `events.jsonl`에 **append-only**로
  기록합니다 (동시 읽기·쓰기 안전).
- 추론 워커는 동일한 포맷으로 `image`, `progress`, `completed` 등을 찍습니다.
- API의 `tail_jsonl()`은 파일의 바이트 오프셋을 유지하며 폴링(`0.25s`),
  완료된 JSON 라인만 yield합니다. Terminal 이벤트(`completed` / `error` /
  `cancelled`)를 만나면 스트림을 닫습니다.
- 유휴 구간에는 15초마다 `__heartbeat__`를 주기적으로 yield → `: heartbeat`
  SSE 주석 프레임으로 변환되어 프록시 타임아웃을 막습니다.

### 3.4 상태 파일의 원자성

`state.json`은 폴링 API(`GET /api/train/{id}`)와 관리자(reaper)가 동시에
쓰기 때문에, `write_json_atomic()`이 tempfile → fsync → `os.replace()`
시퀀스로 쓰기를 원자화합니다. Reader는 언제 읽어도 완전한 JSON을 봅니다.

### 3.5 Reconcile + Reaper

- **`reconcile_on_startup()`** — 서버 기동 시 이전에 `running`으로 남아있던
  job을 찾아 PID가 살아있는지 확인하고, 죽었으면 `failed`로 내려줍니다.
  (개발 중 서버 재시작으로 인한 좀비 상태 방지.)
- **Background reaper** — `app.lifespan`에서 2초 주기로
  `JobManager.reap_finished()` + `InferenceManager.reap_finished()`를
  호출해 종료된 subprocess의 상태를 최신화합니다. 요청이 들어오지
  않아도 `state.json`이 뒤처지지 않습니다.

### 3.6 "모델 = 완료된 job의 output 디렉토리"

별도 모델 레지스트리를 두지 않고, `<JOBS_DIR>/<id>/output/`에 무엇이
들어있는지로 판별합니다.

- `pytorch_lora_weights.bin` → `kind: "lora"`
- `model_index.json` → `kind: "full"`
- 그 외 → 모델 목록에서 제외

추론 워커는 `kind == "lora"`이면 베이스 모델(`DREAMBOOTH_DEFAULT_PRETRAINED`)
위에 LoRA 가중치를 로드하고, `"full"`이면 해당 디렉토리를 그대로
`StableDiffusionPipeline.from_pretrained()`으로 불러옵니다.

### 3.7 경로 안전

`datasets/{id}`, `train/{id}`, `inference/{id}/images/{filename}` 등
사용자가 제어 가능한 경로 조각은 모두 `safe_component()` /
`safe_filename()` 검증을 거쳐 `resolve_under(base, ...)`이 base를
벗어나는지 재확인합니다. path traversal은 `UnsafePathError`로 400.

---

## 4. 데이터 저장소 레이아웃

```
backend/api/data/                           (DREAMBOOTH_API_DATA_ROOT)
├── datasets/
│   └── <dataset_id>/
│       ├── meta.json                       # {id, name, created_at}
│       └── images/                         # 업로드된 원본 이미지
├── jobs/
│   └── <job_id>/
│       ├── state.json                      # 원자적 저장 (JobState)
│       ├── config.json                     # 워커에 넘기는 DreamBoothConfig
│       ├── events.jsonl                    # 학습 이벤트 (SSE 소스)
│       ├── worker.log                      # 워커의 stdout/stderr
│       └── output/                         # 학습 산출물 (= "모델")
│           ├── pytorch_lora_weights.bin    # LoRA인 경우
│           └── model_index.json            # full 파이프라인인 경우
├── inference/
│   └── <infer_job_id>/
│       ├── state.json                      # InferJobState
│       ├── request.json                    # 워커에 넘기는 InferRequest
│       ├── events.jsonl                    # 추론 이벤트
│       ├── worker.log
│       └── output/*.png                    # 생성된 이미지
└── models/                                 # (예약; 현재 미사용)
```

---

## 5. 주요 플로우 요약

### 5.1 학습 시작 (happy path)

1. 브라우저 `TrainNew` → `POST /api/train` (JSON body)
2. `routes/train.start_training`
   - `require_gpu_free()` 통과
   - `_build_training_config()` — dataset 경로 확인, `resolution % 8 == 0`
     검증, preset 기본값 overlay
   - `JobManager.start(cfg)` — `job_id` 발급, `config.json` 저장,
     `train_worker`를 subprocess로 spawn, `state.json`에 `running` 기록
3. 워커: `DreamBoothTrainer(config, event_log_path).train()`
   - `events.jsonl`에 `start` → `step`×N → `validation` → `checkpoint`
     → `completed` 순서로 JSONL 기록
4. 브라우저 `TrainLive` → `EventSource /api/train/{id}/events`
   - `tail_jsonl()`이 레코드를 SSE 프레임으로 흘려보냄
   - `step` 레코드는 `JobManager.update_state()`로 `latest_step/loss` 갱신
   - `completed` 받으면 스트림 종료 → 클라이언트 stream close
5. 백그라운드 reaper가 subprocess 종료를 감지하고 `state.json.status`를
   `completed`로 마감, `finished_at` 기록

### 5.2 이미지 생성

1. `POST /api/inference/generate` with `{model_id, prompts[...]}`
2. `resolve_model(model_id)` — `state.status == completed`이고
   `pytorch_lora_weights.bin` 또는 `model_index.json`이 있어야 함
3. `InferenceManager.start(...)` — `infer_worker`를 spawn
4. 워커: 프롬프트별 이미지를 생성, 각 완료 시 `events.jsonl`에
   `{"type":"image","filename":"..."}` 기록
5. 프런트엔드 `Generate`는 SSE로 `image` 이벤트를 받는 즉시
   `/api/inference/{id}/images/{filename}` URL을 `<img>`로 렌더링

### 5.3 중단

- `POST /api/train|inference/{id}/stop`
- 매니저가 프로세스 그룹에 `SIGTERM` → 5초(학습) / 3초(추론) 대기 →
  살아있으면 `SIGKILL`
- `_finalize(cancelled=True)` → `state.status = "cancelled"`,
  `events.jsonl`에 `cancelled` 이벤트가 남아있으면 SSE가 깔끔히 종료

---

## 6. 8GB VRAM 최적화 체크리스트

| 항목 | 위치 |
|------|------|
| Batch size 1 | `DreamBoothConfig.train_batch_size = 1` |
| Gradient accumulation | `train_batch_size=1 + grad_accum=4` (non-LoRA), `=1` (LoRA) |
| 8-bit Adam | `use_8bit_adam=True` (bitsandbytes) |
| Mixed precision FP16 | `mixed_precision="fp16"` |
| Gradient checkpointing | `gradient_checkpointing=True` |
| xformers (옵션) | 기본 우회, `DREAMBOOTH_ENABLE_XFORMERS=1`로 활성화 |
| VAE slicing | `enable_vae_slicing=True` |
| VAE tiling (큰 해상도) | `enable_vae_tiling=True` |
| Text encoder CPU offload | `cpu_offload_text_encoder=True` |
| LoRA fine-tuning | `use_lora=True`, `lora_rank=4` |
| 동시 GPU 작업 1개 | `concurrency.require_gpu_free()` |

---

## 7. 확장 포인트

- **다중 GPU / 다중 작업**: `concurrency.require_gpu_free()`를 device
  지정 큐로 교체하면, 워커에 `CUDA_VISIBLE_DEVICES`를 다르게 주입하는
  것으로 확장 가능. 현재는 의도적으로 비활성화.
- **원격 배포**: CORS 화이트리스트(`DREAMBOOTH_CORS_ORIGINS`), 업로드
  크기(`DREAMBOOTH_MAX_UPLOAD_BYTES`)가 환경 변수로 노출돼 있음.
- **모델 레지스트리**: 현재는 "완료된 job == 모델"로 취급. 외부에서
  다운로드한 모델을 등록하려면 `routes/models.py`에 import 경로를
  추가.
- **멀티테넌시**: `paths.safe_component()` / `resolve_under()`가 이미
  path traversal을 방지하므로, 인증 레이어만 얹으면 사용자별 `DATA_ROOT`
  분리가 쉽다.

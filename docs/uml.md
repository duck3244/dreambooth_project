# UML Diagrams

모든 다이어그램은 Mermaid로 작성했습니다. GitHub, VS Code, Typora 등
Mermaid 렌더링을 지원하는 뷰어에서 그대로 열어볼 수 있습니다.

- [1. 컴포넌트(패키지) 다이어그램](#1-컴포넌트패키지-다이어그램)
- [2. 클래스 다이어그램 — 학습/추론 코어](#2-클래스-다이어그램--학습추론-코어)
- [3. 클래스 다이어그램 — API 레이어](#3-클래스-다이어그램--api-레이어)
- [4. 시퀀스 다이어그램 — 학습 시작부터 완료까지](#4-시퀀스-다이어그램--학습-시작부터-완료까지)
- [5. 시퀀스 다이어그램 — 이미지 생성 + SSE 스트리밍](#5-시퀀스-다이어그램--이미지-생성--sse-스트리밍)
- [6. 시퀀스 다이어그램 — 학습 중단](#6-시퀀스-다이어그램--학습-중단)
- [7. 상태 다이어그램 — Job lifecycle](#7-상태-다이어그램--job-lifecycle)
- [8. 플로우차트 — CLI 실행 모드](#8-플로우차트--cli-실행-모드)

---

## 1. 컴포넌트(패키지) 다이어그램

```mermaid
graph TB
    subgraph Browser["Browser (Vite :5173)"]
        Pages["React Pages<br/>Dashboard · Datasets · TrainNew<br/>TrainLive · Models · Generate"]
        ApiTs["api.ts / types.ts"]
        UseSSE["hooks/useSSE.ts"]
    end

    subgraph API["FastAPI (uvicorn :8765)  —  no torch"]
        App["app.py<br/>create_app + lifespan"]
        subgraph Routes["routes/"]
            RHealth[health]
            RGpu[gpu]
            RDatasets[datasets]
            RTrain[train]
            RModels[models]
            RInference[inference]
        end
        Concurrency[concurrency<br/>require_gpu_free]
        JM[JobManager]
        IM[InferenceManager]
        EventLog[event_log<br/>tail_jsonl / sse_frame]
        State[state<br/>write_json_atomic]
        GpuMod[gpu<br/>pynvml]
        Settings[settings]
        Paths[paths]
    end

    subgraph Workers["subprocess workers  (GPU 소유)"]
        TrainW[workers/train_worker.py]
        InferW[workers/infer_worker.py]
    end

    subgraph Core["학습/추론 코어 (backend/*.py)"]
        Config[config.py<br/>DreamBoothConfig + PresetConfigs]
        Dataset[dataset.py<br/>DreamBoothDataset]
        Model[model.py<br/>MemoryOptimizedModel + PipelineManager]
        Trainer[trainer.py<br/>DreamBoothTrainer]
        Utils[utils.py]
        MainCli["main.py (CLI)"]
    end

    FS[("data/<br/>datasets/  jobs/  inference/")]
    GPU((RTX 4060 8GB))

    Pages --> ApiTs
    Pages --> UseSSE
    ApiTs -->|HTTP /api/*| Routes
    UseSSE -->|SSE events| Routes

    Routes --> Concurrency
    Routes --> JM
    Routes --> IM
    Routes --> EventLog
    Routes --> State
    Routes --> GpuMod
    App --> JM
    App --> IM

    JM -->|subprocess.Popen| TrainW
    IM -->|subprocess.Popen| InferW

    TrainW --> Config
    TrainW --> Trainer
    InferW --> Model
    Trainer --> Model
    Trainer --> Dataset
    Trainer --> Config
    MainCli --> Config
    MainCli --> Trainer
    MainCli --> Utils

    JM --> FS
    IM --> FS
    TrainW --> FS
    InferW --> FS

    TrainW --> GPU
    InferW --> GPU
```

---

## 2. 클래스 다이어그램 — 학습/추론 코어

`backend/config.py`, `dataset.py`, `model.py`, `trainer.py`, `utils.py`.

```mermaid
classDiagram
    class DreamBoothConfig {
      +str pretrained_model_name_or_path
      +str instance_data_dir
      +str class_data_dir
      +str output_dir
      +str instance_prompt
      +str class_prompt
      +int resolution
      +int train_batch_size
      +int gradient_accumulation_steps
      +float learning_rate
      +str lr_scheduler
      +int max_train_steps
      +int validation_steps
      +int checkpointing_steps
      +bool use_8bit_adam
      +bool gradient_checkpointing
      +str mixed_precision
      +bool enable_vae_slicing
      +bool enable_vae_tiling
      +bool cpu_offload_text_encoder
      +bool use_lora
      +int lora_rank
      +int lora_alpha
      +tuple lora_target_modules
      +bool with_prior_preservation
      +int seed
      +bool deterministic
      +__post_init__()
      +to_dict() dict
      +get_env_vars() dict
    }

    class PresetConfigs {
      <<static>>
      +get_person_config() DreamBoothConfig
      +get_object_config() DreamBoothConfig
      +get_style_config() DreamBoothConfig
      +get_fast_config() DreamBoothConfig
      +get_high_quality_config() DreamBoothConfig
    }

    class DreamBoothDataset {
      +Path instance_data_root
      +str instance_prompt
      +Tokenizer tokenizer
      +int size
      +_collect_images() List~Path~
      +_create_transforms() Compose
      +_load_and_preprocess_image(path) Tensor
      +_tokenize_prompt(prompt) Tensor
      +__len__() int
      +__getitem__(i) dict
    }

    class ImageValidator {
      <<static>>
      +validate_image(path) Tuple~bool,str~
      +validate_dataset(root) dict
    }

    class DatasetUtils {
      <<static>>
      +create_dataloader(ds, bs, shuffle) DataLoader
      +preview_dataset(ds, n)
      +analyze_dataset(root) dict
    }

    class MemoryOptimizedModel {
      +str model_name
      +torch.device device
      +CLIPTokenizer tokenizer
      +CLIPTextModel text_encoder
      +AutoencoderKL vae
      +UNet2DConditionModel unet
      +DDPMScheduler scheduler
      +load_components(load_vae)
      +enable_memory_optimization()
      +enable_vae_optimizations(slicing, tiling)
      +cast_frozen_components(dtype)
      +apply_lora(rank, alpha, targets, dropout)
      +prepare_for_training()
      +get_trainable_parameters() List
      +save_checkpoint(dir, step)
      +load_checkpoint(path) int
      +create_pipeline(dir) StableDiffusionPipeline
      +clear_memory()
    }

    class ModelOptimizer {
      <<static>>
      +get_8bit_adam_optimizer(params, lr) Optimizer
      +calculate_memory_usage() dict
      +optimize_for_8gb_vram(model)
    }

    class ModelValidator {
      <<static>>
      +validate_model_components(model) dict
      +test_forward_pass(model, bs) bool
    }

    class PipelineManager {
      +str model_path
      +StableDiffusionPipeline pipeline
      +load_pipeline() StableDiffusionPipeline
      +generate_image(prompt, ...) Image
      +batch_generate(prompts, ...) List
    }

    class DreamBoothTrainer {
      +DreamBoothConfig config
      +Accelerator accelerator
      +MemoryOptimizedModel model
      +Optimizer optimizer
      +LRScheduler lr_scheduler
      +int global_step
      +str event_log_path
      +setup_logging()
      +setup_reproducibility()
      +setup_model()
      +setup_optimizer_and_scheduler()
      +setup_dataset()
      +prepare_accelerator()
      +compute_loss(batch) Tensor
      +training_step(batch) float
      +validate()
      +generate_validation_images() list
      +save_checkpoint()
      +cleanup_old_checkpoints()
      +train()
      +save_final_model()
      +resume_from_checkpoint(path)
      +emit_event(type, **fields)
      +close_event_log()
    }

    class ImageProcessor {
      <<static>>
      +resize_and_crop(img, size) Image
      +enhance_image(img, brightness, ...) Image
      +remove_background(src, dst) bool
      +batch_process_images(in, out, size)
    }

    class SystemMonitor {
      <<static>>
      +get_system_info() dict
      +monitor_training(log, interval)
      +plot_training_metrics(log, dir)
    }

    class ModelTester {
      +str model_path
      +StableDiffusionPipeline pipeline
      +load_pipeline() bool
      +test_generation(prompts, dir) bool
      +benchmark_performance(prompt, n) dict
    }

    class ProjectSetup {
      <<static>>
      +create_project_structure(name) str
      +check_requirements() dict
      +install_requirements() str
    }

    class ConfigValidator {
      <<static>>
      +validate_training_setup(config) Tuple
    }

    PresetConfigs ..> DreamBoothConfig : creates
    DreamBoothTrainer --> DreamBoothConfig : uses
    DreamBoothTrainer --> MemoryOptimizedModel : owns
    DreamBoothTrainer --> DreamBoothDataset : owns
    DreamBoothTrainer --> ModelOptimizer : uses
    DreamBoothTrainer --> PipelineManager : uses (validation)
    MemoryOptimizedModel ..> ModelValidator : validated by
    MemoryOptimizedModel --> PipelineManager : creates
    DreamBoothDataset ..> ImageValidator : validated by
    DreamBoothDataset ..> DatasetUtils : utility
    ModelTester --> PipelineManager : wraps
    ModelTester ..> SystemMonitor : uses
```

---

## 3. 클래스 다이어그램 — API 레이어

`backend/api/**`. Pydantic 스키마는 대표적인 것만 표기했습니다.

```mermaid
classDiagram
    class FastAPIApp {
      +create_app() FastAPI
      +lifespan(app)
    }

    class Settings {
      <<module>>
      +Path BACKEND_DIR
      +Path DATA_ROOT
      +Path DATASETS_DIR
      +Path JOBS_DIR
      +Path MODELS_DIR
      +Path INFERENCE_DIR
      +str DEFAULT_PRETRAINED
      +List CORS_ORIGINS
      +int MAX_UPLOAD_BYTES
      +tuple ALLOWED_IMAGE_SUFFIXES
      +ensure_dirs()
    }

    class Paths {
      <<module>>
      +safe_component(name) str
      +safe_filename(name) str
      +resolve_under(base, *parts) Path
    }

    class UnsafePathError

    class StateIO {
      <<module>>
      +read_json(path, default) Any
      +write_json_atomic(path, data)
    }

    class EventLog {
      <<module>>
      +sse_frame(data, event) str
      +tail_jsonl(path, poll, heartbeat, is_alive) AsyncIterator
    }

    class Concurrency {
      <<module>>
      +active_gpu_job(request) Optional
      +require_gpu_free(request)
    }

    class GpuMod {
      <<module>>
      +init()
      +shutdown()
      +get_info(index) GPUInfo
    }

    class JobManager {
      -Path jobs_dir
      -Dict _procs
      -Lock _lock
      +job_dir(id) Path
      +event_log_path(id) Path
      +output_dir(id) Path
      +read_state(id) dict
      +write_state(id, state)
      +update_state(id, **fields) dict
      +start(cfg, dataset_id) dict
      +is_alive(id) bool
      +has_running_job() Optional~str~
      +stop(id, grace) dict
      +reap_finished()
      +list_jobs() List
      +reconcile_on_startup()
    }

    class InferenceManager {
      -Path base_dir
      -Dict _procs
      -Lock _lock
      +job_dir(id) Path
      +output_dir(id) Path
      +start(model_id, model_path, prompts, ...) dict
      +is_alive(id) bool
      +has_running_job() Optional~str~
      +stop(id, grace) dict
      +reap_finished()
      +_list_output_images(id) List
      +list_jobs() List
      +reconcile_on_startup()
    }

    class _RunningProc {
      +Popen popen
      +IO stdout_fp
    }

    class TrainRoutes {
      <<router>>
      +POST /api/train
      +GET  /api/train
      +GET  /api/train/:id
      +DELETE /api/train/:id
      +POST /api/train/:id/stop
      +GET  /api/train/:id/events  -- SSE
    }

    class InferenceRoutes {
      <<router>>
      +POST /api/inference/generate
      +GET  /api/inference
      +GET  /api/inference/:id
      +DELETE /api/inference/:id
      +POST /api/inference/:id/stop
      +GET  /api/inference/:id/events  -- SSE
      +GET  /api/inference/:id/images/:filename
    }

    class DatasetRoutes {
      <<router>>
      +GET /api/datasets
      +POST /api/datasets
      +GET /api/datasets/:id
      +POST /api/datasets/:id/images
      +DELETE /api/datasets/:id
    }

    class ModelsRoutes {
      <<router>>
      +GET /api/models
      +resolve_model(id) ModelInfo?
    }

    class TrainWorker {
      <<subprocess>>
      +main()
    }

    class InferWorker {
      <<subprocess>>
      +main()
    }

    class TrainStartRequest {
      +str dataset_id
      +str preset
      +str instance_prompt
      +str? class_prompt
      +int max_train_steps
      +float learning_rate
      +int resolution
      +bool use_lora
      +int lora_rank / lora_alpha
      +str mixed_precision
      +bool enable_vae_slicing / tiling
      +bool cpu_offload_text_encoder
      +bool with_prior_preservation
      +int seed
      +bool deterministic
    }

    class JobState {
      +str id
      +str status
      +float created_at / started_at / finished_at
      +int? pid / return_code
      +str? error
      +str? dataset_id / output_dir / event_log
      +int? max_train_steps / latest_step
      +float? latest_loss
    }

    class InferRequest {
      +str model_id
      +List~str~ prompts
      +str? negative_prompt
      +int num_inference_steps
      +float guidance_scale
      +int height / width
      +int num_images_per_prompt
      +int? seed
    }

    class InferJobState {
      +str id
      +str status
      +str model_id
      +float created_at / started_at / finished_at
      +List~str~ prompts
      +List~str~ images
      +int total_images
    }

    class ModelInfo {
      +str id
      +str? job_id
      +str path
      +str kind  (lora | full)
      +float created_at
    }

    FastAPIApp --> JobManager : owns (app.state)
    FastAPIApp --> InferenceManager : owns (app.state)
    FastAPIApp --> GpuMod : init/shutdown
    FastAPIApp --> Settings : ensure_dirs

    JobManager --> _RunningProc
    InferenceManager --> _RunningProc
    JobManager ..> StateIO : read/write
    InferenceManager ..> StateIO : read/write
    JobManager ..> Paths
    InferenceManager ..> Paths
    JobManager ..> Settings
    InferenceManager ..> Settings

    JobManager --> TrainWorker : spawns
    InferenceManager --> InferWorker : spawns

    TrainRoutes --> JobManager
    TrainRoutes ..> Concurrency
    TrainRoutes ..> EventLog : SSE
    TrainRoutes ..> TrainStartRequest
    TrainRoutes ..> JobState

    InferenceRoutes --> InferenceManager
    InferenceRoutes ..> Concurrency
    InferenceRoutes ..> EventLog : SSE
    InferenceRoutes ..> ModelsRoutes : resolve_model
    InferenceRoutes ..> InferRequest
    InferenceRoutes ..> InferJobState

    DatasetRoutes ..> Paths
    DatasetRoutes ..> StateIO
    DatasetRoutes ..> Settings

    ModelsRoutes ..> StateIO
    ModelsRoutes ..> Settings
    ModelsRoutes ..> ModelInfo

    Paths ..> UnsafePathError
```

---

## 4. 시퀀스 다이어그램 — 학습 시작부터 완료까지

```mermaid
sequenceDiagram
    autonumber
    actor U as User (Browser)
    participant Web as React (TrainNew / TrainLive)
    participant API as FastAPI routes/train
    participant JM as JobManager
    participant FS as state.json / events.jsonl
    participant W as train_worker (subprocess)
    participant Core as DreamBoothTrainer + Model
    participant GPU as RTX 4060

    U->>Web: 파라미터 입력 후 Start
    Web->>API: POST /api/train (TrainStartRequest)
    API->>API: require_gpu_free()
    API->>API: _build_training_config()<br/>(dataset 검증, resolution%8)
    API->>JM: start(cfg, dataset_id)
    JM->>FS: write state.json {status: running, pid}
    JM->>W: subprocess.Popen(python -m train_worker<br/>--config --event-log,<br/>start_new_session=True)
    JM-->>API: initial JobState
    API-->>Web: 201 JobState

    Web->>Web: navigate /train/{id}
    Web->>API: EventSource /api/train/{id}/events

    par 학습 루프
        W->>Core: load config → setup_model / dataset / optimizer
        Core->>GPU: UNet / VAE / TextEncoder load
        loop training steps
            Core->>GPU: forward + backward + step
            W->>FS: append {"type":"step", step, loss}
        end
        Core->>GPU: validation image generation
        W->>FS: append {"type":"validation", ...}
        Core->>FS: save checkpoint
        W->>FS: append {"type":"completed"}
    and SSE 스트림
        loop tail_jsonl
            API->>FS: stat + read new bytes
            FS-->>API: JSONL records
            API->>JM: update_state(latest_step, latest_loss)<br/>(type=step 일 때)
            API-->>Web: SSE frame (event: step/validation/...)
        end
        API-->>Web: SSE "completed" then stream close
    end

    Note over JM,FS: 백그라운드 reaper가 2초 주기로<br/>reap_finished() → state.status = completed
    Web->>API: GET /api/train/{id}
    API-->>Web: JobState (status=completed, output_dir)
    Web->>API: GET /api/models
    API-->>Web: ModelList (kind: lora | full)
```

---

## 5. 시퀀스 다이어그램 — 이미지 생성 + SSE 스트리밍

```mermaid
sequenceDiagram
    autonumber
    actor U as User (Browser)
    participant Web as React (Generate)
    participant API as routes/inference
    participant IM as InferenceManager
    participant MR as routes/models.resolve_model
    participant FS as state.json / events.jsonl / output/
    participant W as infer_worker (subprocess)
    participant GPU as RTX 4060

    U->>Web: 모델 선택 + 프롬프트 입력
    Web->>API: POST /api/inference/generate (InferRequest)
    API->>API: require_gpu_free()
    API->>MR: resolve_model(model_id)
    MR-->>API: ModelInfo(kind, path) or 404
    API->>IM: start(model_path, kind, prompts, ...)
    IM->>FS: write request.json + state.json {running}
    IM->>W: Popen(python -m infer_worker)
    IM-->>API: initial InferJobState
    API-->>Web: 201 InferJobState

    Web->>API: EventSource /api/inference/{id}/events

    par 추론 실행
        W->>GPU: StableDiffusionPipeline.from_pretrained(...)
        alt kind == "lora"
            W->>GPU: pipeline.load_lora_weights(model_path)
        end
        loop prompts × num_images_per_prompt
            W->>GPU: pipeline(prompt, steps, guidance, ...)
            W->>FS: save PNG to output/
            W->>FS: append {"type":"image", filename, prompt_index}
        end
        W->>FS: append {"type":"completed", total}
    and SSE tail
        loop tail_jsonl
            API->>FS: read new JSONL
            API-->>Web: SSE frame (event: image / progress / completed)
        end
    end

    loop on each "image" event
        Web->>API: GET /api/inference/{id}/images/{filename}
        API-->>Web: PNG (FileResponse)
        Web-->>U: <img> 렌더링
    end

    API-->>Web: SSE completed → stream close
```

---

## 6. 시퀀스 다이어그램 — 학습 중단

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant Web
    participant API as routes/train.stop_job
    participant JM as JobManager
    participant W as train_worker pgrp
    participant FS as state.json

    U->>Web: Stop 버튼
    Web->>API: POST /api/train/{id}/stop
    API->>JM: stop(id, grace=5.0)
    JM->>W: os.killpg(pgid, SIGTERM)
    loop up to grace seconds
        JM->>W: poll()
    end
    alt still alive
        JM->>W: SIGKILL
    end
    JM->>FS: _finalize(cancelled=True)<br/>status=cancelled, return_code
    JM-->>API: JobState
    API-->>Web: JobState(status=cancelled)
    Note over Web: TrainLive의 SSE는 "cancelled"를 받고<br/>stream을 닫는다.
```

---

## 7. 상태 다이어그램 — Job lifecycle

학습 / 추론 모두 동일한 상태 머신을 씁니다 (`schemas.JobState.status`,
`InferJobState.status`).

```mermaid
stateDiagram-v2
    [*] --> pending : start() 직후 (거의 순간 transition)
    pending --> running : Popen 성공, pid 기록
    running --> completed : worker exit code 0<br/>+ events.jsonl "completed"
    running --> failed    : worker exit code ≠ 0<br/>or "error" event
    running --> cancelled : stop() (SIGTERM → SIGKILL)<br/>or worker 사망 감지
    completed --> [*]
    failed --> [*]
    cancelled --> [*]

    note right of running
      reconcile_on_startup():
        서버 재시작 시 pid 생존 확인,
        죽었으면 failed 로 이동.

      background reaper:
        2초마다 reap_finished().
    end note
```

---

## 8. 플로우차트 — CLI 실행 모드

API를 거치지 않고 `python backend/main.py --mode ...`로 직접 실행할 때의
흐름입니다.

```mermaid
flowchart TD
    Start([python main.py]) --> Parse[argparse]
    Parse --> Mode{--mode}

    Mode -->|setup| S1[ProjectSetup.create_project_structure]
    S1 --> S2[check_requirements / system info]
    S2 --> End([종료])

    Mode -->|validate| V1[create_config_from_args]
    V1 --> V2[ConfigValidator.validate_training_setup]
    V2 --> V3[ImageValidator.validate_dataset]
    V3 --> V4{통과?}
    V4 -->|Yes| End
    V4 -->|No| VE[에러 출력] --> End

    Mode -->|train| T1[setup_environment]
    T1 --> T2[create_config_from_args<br/>+ PresetConfigs overlay]
    T2 --> T3[process_images_if_needed]
    T3 --> T4[DreamBoothTrainer = config]
    T4 --> T5[setup_model / optimizer / dataset]
    T5 --> T6[prepare_accelerator]
    T6 --> T7[loop: training_step → validate → checkpoint]
    T7 --> T8{global_step &ge; max?}
    T8 -->|No| T7
    T8 -->|Yes| T9[save_final_model]
    T9 --> End

    Mode -->|test| M1[ModelTester model_path]
    M1 --> M2[load_pipeline]
    M2 --> M3[test_generation prompts]
    M3 --> M4[benchmark_performance]
    M4 --> End
```

---

## 부록 — 이벤트 JSONL 스키마

워커가 `events.jsonl`에 append하는 레코드의 `type` 필드별 페이로드입니다.

| type | 필드 | 설명 |
|------|------|------|
| `start` | `ts`, `config` | 학습 시작. API는 프런트 초기화에 사용 |
| `step` | `ts`, `step`, `loss`, `lr`, `eta_seconds?` | 학습 스텝. API가 `JobManager.update_state()`로 `latest_step/loss` 갱신 |
| `validation` | `ts`, `step`, `images?` | 검증 이미지 생성 |
| `checkpoint` | `ts`, `step`, `path` | 체크포인트 저장 |
| `image` (추론) | `ts`, `filename`, `prompt_index`, `prompt` | 이미지 하나 생성 완료 |
| `progress` (추론) | `ts`, `done`, `total` | 전체 진행률 |
| `completed` | `ts`, … | **terminal** — SSE 스트림 종료 |
| `error` | `ts`, `error`, `error_type` | **terminal** |
| `cancelled` | `ts`, `reason?` | **terminal** — stop() 호출 또는 워커 비정상 종료 감지 |
| `__heartbeat__` | (내부용) | 15초 유휴 시 API가 생성. SSE 주석 프레임으로 변환되어 프록시 keepalive |

`_TERMINAL_TYPES = {"completed", "error", "cancelled"}` — `tail_jsonl`은
이 중 하나를 만나면 스트림을 닫아 `EventSource`가 재연결을 시도하지
않도록 합니다.

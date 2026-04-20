"""Pydantic request/response schemas for the HTTP layer."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- GPU ----------

class GPUInfo(BaseModel):
    available: bool
    name: Optional[str] = None
    driver_version: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    utilization_percent: Optional[int] = None
    temperature_c: Optional[int] = None
    error: Optional[str] = None


# ---------- Datasets ----------

class DatasetInfo(BaseModel):
    id: str
    name: str
    image_count: int
    total_bytes: int
    created_at: float


class DatasetList(BaseModel):
    datasets: List[DatasetInfo]


class DatasetCreateResponse(BaseModel):
    id: str
    name: str


class DatasetUploadResponse(BaseModel):
    id: str
    added: List[str]
    skipped: List[str]
    image_count: int


# ---------- Training ----------

Preset = Literal["person", "object", "style", "fast", "high_quality"]


class TrainStartRequest(BaseModel):
    dataset_id: str = Field(..., description="Registered dataset id")
    preset: Preset = "person"
    instance_prompt: str = "a photo of sks person"
    class_prompt: Optional[str] = None
    max_train_steps: int = Field(400, ge=1, le=5000)
    learning_rate: float = Field(1e-6, gt=0, le=1e-3)
    resolution: int = Field(512, ge=64, le=1024)
    use_lora: bool = True
    lora_rank: int = Field(4, ge=1, le=128)
    lora_alpha: int = Field(4, ge=1, le=128)
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    cpu_offload_text_encoder: bool = False
    with_prior_preservation: bool = False
    seed: int = 42
    deterministic: bool = False
    pretrained_model_name_or_path: Optional[str] = None


class JobState(BaseModel):
    id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    dataset_id: Optional[str] = None
    output_dir: Optional[str] = None
    event_log: Optional[str] = None
    max_train_steps: Optional[int] = None
    latest_step: Optional[int] = None
    latest_loss: Optional[float] = None


class JobList(BaseModel):
    jobs: List[JobState]


# ---------- Models ----------

class ModelInfo(BaseModel):
    id: str
    job_id: Optional[str] = None
    path: str
    kind: Literal["lora", "full"]
    created_at: float


class ModelList(BaseModel):
    models: List[ModelInfo]


# ---------- Inference ----------

class InferRequest(BaseModel):
    model_id: str = Field(..., description="Model id from /api/models")
    prompts: List[str] = Field(..., min_length=1, max_length=8)
    negative_prompt: Optional[str] = None
    num_inference_steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=0.0, le=30.0)
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    num_images_per_prompt: int = Field(1, ge=1, le=4)
    seed: Optional[int] = None


class InferJobState(BaseModel):
    id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    model_id: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    event_log: Optional[str] = None
    prompts: List[str] = []
    images: List[str] = Field(default_factory=list)
    total_images: int = 0


class InferJobList(BaseModel):
    jobs: List[InferJobState]


# ---------- Generic ----------

class OkResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

"""API-layer runtime settings.

Keep this module dependency-free (no torch, no heavy imports) so the API
process starts in sub-second. All paths are env-overridable for tests and
deployment.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).resolve() if raw else default.resolve()


# Project layout anchors
BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Path = BACKEND_DIR.parent

# Data root for API-owned state (jobs, uploaded datasets, logs, outputs).
# Separate from backend/instance_images + backend/dreambooth_output, which
# remain CLI-compatible.
DATA_ROOT: Path = _env_path("DREAMBOOTH_API_DATA_ROOT", BACKEND_DIR / "api" / "data")

DATASETS_DIR: Path = DATA_ROOT / "datasets"
JOBS_DIR: Path = DATA_ROOT / "jobs"
MODELS_DIR: Path = DATA_ROOT / "models"
INFERENCE_DIR: Path = DATA_ROOT / "inference"

# Worker entry modules (invoked via `python -m`)
TRAIN_WORKER_MODULE = "backend.api.workers.train_worker"
INFER_WORKER_MODULE = "backend.api.workers.infer_worker"

# Default pretrained checkpoint for API-initiated jobs (can be overridden per request).
DEFAULT_PRETRAINED = os.environ.get(
    "DREAMBOOTH_DEFAULT_PRETRAINED",
    "CompVis/stable-diffusion-v1-4",
)

# CORS origins for the dev frontend.
CORS_ORIGINS = os.environ.get(
    "DREAMBOOTH_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# Dataset upload limits
MAX_UPLOAD_BYTES = int(os.environ.get("DREAMBOOTH_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50MB
ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def ensure_dirs() -> None:
    for p in (DATA_ROOT, DATASETS_DIR, JOBS_DIR, MODELS_DIR, INFERENCE_DIR):
        p.mkdir(parents=True, exist_ok=True)

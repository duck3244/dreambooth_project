"""Training job routes."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .. import settings
from ..concurrency import require_gpu_free
from ..event_log import sse_frame, tail_jsonl
from ..job_manager import JobManager
from ..paths import UnsafePathError, safe_component
from ..schemas import JobList, JobState, OkResponse, TrainStartRequest
from ..state import read_json
from .datasets import dataset_images_path


router = APIRouter(prefix="/train", tags=["train"])


def _manager(request: Request) -> JobManager:
    return request.app.state.job_manager


MIN_INSTANCE_IMAGES = 3


def _build_training_config(req: TrainStartRequest) -> dict:
    """Map the API request to a DreamBoothConfig-compatible dict.

    Only whitelisted fields — any extras are dropped in the worker anyway.
    """
    # Resolve dataset → filesystem path (also validates the id).
    try:
        images_path = dataset_images_path(req.dataset_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid dataset_id: {e}")
    if not images_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"dataset {req.dataset_id} has no images dir")

    # Pre-flight: need enough images for meaningful fine-tuning.
    image_count = sum(
        1 for p in images_path.iterdir()
        if p.is_file() and p.suffix.lower() in settings.ALLOWED_IMAGE_SUFFIXES
    )
    if image_count < MIN_INSTANCE_IMAGES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"dataset has {image_count} images; need at least {MIN_INSTANCE_IMAGES}",
        )

    if req.with_prior_preservation and not (req.class_prompt and req.class_prompt.strip()):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "class_prompt is required when with_prior_preservation is true",
        )

    if req.resolution % 8 != 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "resolution must be a multiple of 8")

    preset_max_steps = {
        "person": 400, "object": 600, "style": 800, "fast": 200, "high_quality": 800,
    }
    # Start from the preset-implied defaults, then overlay the request.
    cfg: dict = {
        "pretrained_model_name_or_path": req.pretrained_model_name_or_path or settings.DEFAULT_PRETRAINED,
        "instance_data_dir": str(images_path),
        "instance_prompt": req.instance_prompt,
        "class_prompt": req.class_prompt or "a photo of person",
        "resolution": req.resolution,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4 if not req.use_lora else 1,
        "learning_rate": req.learning_rate,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_train_steps": req.max_train_steps,
        "validation_prompt": req.instance_prompt,
        "validation_steps": max(100, req.max_train_steps // 4) if req.max_train_steps >= 100 else req.max_train_steps,
        "checkpointing_steps": max(100, req.max_train_steps // 4) if req.max_train_steps >= 100 else req.max_train_steps,
        "mixed_precision": req.mixed_precision,
        "use_8bit_adam": True,
        "gradient_checkpointing": True,
        "enable_vae_slicing": req.enable_vae_slicing,
        "enable_vae_tiling": req.enable_vae_tiling,
        "cpu_offload_text_encoder": req.cpu_offload_text_encoder,
        "use_lora": req.use_lora,
        "lora_rank": req.lora_rank,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": 0.0,
        "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "with_prior_preservation": req.with_prior_preservation,
        "seed": req.seed,
        "deterministic": req.deterministic,
        "num_workers": 0,
    }
    # Sanity: preset hint (only applied when request did not override max_train_steps)
    if req.max_train_steps == 400 and req.preset in preset_max_steps:
        cfg["max_train_steps"] = preset_max_steps[req.preset]

    return cfg


@router.post("", response_model=JobState)
async def start_training(req: TrainStartRequest, request: Request) -> JobState:
    require_gpu_free(request)
    cfg = _build_training_config(req)
    manager = _manager(request)
    state = manager.start(cfg, dataset_id=req.dataset_id)
    return JobState(**_normalize_state(state))


@router.get("", response_model=JobList)
async def list_jobs(request: Request) -> JobList:
    manager = _manager(request)
    jobs = [JobState(**_normalize_state(s)) for s in manager.list_jobs()]
    return JobList(jobs=jobs)


@router.get("/{job_id}", response_model=JobState)
async def get_job(job_id: str, request: Request) -> JobState:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    manager.reap_finished()
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return JobState(**_normalize_state(state))


@router.delete("/{job_id}", response_model=OkResponse)
async def delete_job(job_id: str, request: Request) -> OkResponse:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    manager.reap_finished()
    if manager.is_alive(jid):
        raise HTTPException(status.HTTP_409_CONFLICT, "job is still running; stop it first")
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    jd = manager.job_dir(jid)
    if jd.exists():
        shutil.rmtree(jd)
    return OkResponse()


@router.post("/{job_id}/stop", response_model=JobState)
async def stop_job(job_id: str, request: Request) -> JobState:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if not manager.read_state(jid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    state = manager.stop(jid)
    return JobState(**_normalize_state(state))


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")

    event_log = manager.event_log_path(jid)

    async def event_stream():
        # Open comment to flush headers immediately (proxies and browsers).
        yield ": stream open\n\n"
        try:
            async for record in tail_jsonl(
                event_log,
                is_job_alive=lambda: manager.is_alive(jid),
            ):
                # Heartbeats use SSE comment frames (no event), keeping connection alive.
                if record.get("type") == "__heartbeat__":
                    yield ": heartbeat\n\n"
                    continue

                # Update state.json with latest step/loss for quick polling.
                if record.get("type") == "step":
                    manager.update_state(
                        jid,
                        latest_step=record.get("step"),
                        latest_loss=record.get("loss"),
                    )

                yield sse_frame(record, event=record.get("type"))

                # If the client disconnected, stop the tail.
                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield sse_frame({"type": "error", "error": str(e)}, event="error")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable nginx buffering
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _normalize_state(s: dict) -> dict:
    """Ensure the dict conforms to JobState (fill missing optional keys)."""
    fields = {
        "id", "status", "created_at", "started_at", "finished_at", "pid",
        "return_code", "error", "dataset_id", "output_dir", "event_log",
        "max_train_steps", "latest_step", "latest_loss",
    }
    return {k: s.get(k) for k in fields if k in s or k in {"id", "status", "created_at"}}

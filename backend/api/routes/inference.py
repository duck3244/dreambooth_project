"""Inference routes: generate / status / events / serve images."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from .. import settings
from ..concurrency import require_gpu_free
from ..event_log import sse_frame, tail_jsonl
from ..inference_manager import InferenceManager
from ..paths import UnsafePathError, resolve_under, safe_component, safe_filename
from ..schemas import InferJobList, InferJobState, InferRequest, OkResponse
from .models import resolve_model


router = APIRouter(prefix="/inference", tags=["inference"])


def _manager(request: Request) -> InferenceManager:
    return request.app.state.inference_manager


def _normalize_state(s: dict) -> dict:
    fields = {
        "id", "status", "model_id", "created_at", "started_at", "finished_at",
        "pid", "return_code", "error", "output_dir", "event_log", "prompts",
        "images", "total_images",
    }
    out = {k: s.get(k) for k in fields if k in s or k in {"id", "status", "model_id", "created_at"}}
    out.setdefault("prompts", [])
    out.setdefault("images", [])
    out.setdefault("total_images", 0)
    return out


@router.post("/generate", response_model=InferJobState)
async def generate(req: InferRequest, request: Request) -> InferJobState:
    require_gpu_free(request)

    model = resolve_model(req.model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model {req.model_id} not found or not completed")

    for p in req.prompts:
        if not p or not p.strip():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompts must be non-empty")

    if req.height % 8 != 0 or req.width % 8 != 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "height and width must be multiples of 8",
        )

    manager = _manager(request)
    state = manager.start(
        model_id=model.id,
        model_path=model.path,
        model_kind=model.kind,
        base_model=settings.DEFAULT_PRETRAINED,
        prompts=req.prompts,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        height=req.height,
        width=req.width,
        num_images_per_prompt=req.num_images_per_prompt,
        seed=req.seed,
    )
    return InferJobState(**_normalize_state(state))


@router.get("", response_model=InferJobList)
async def list_inference_jobs(request: Request) -> InferJobList:
    manager = _manager(request)
    jobs = [InferJobState(**_normalize_state(s)) for s in manager.list_jobs()]
    return InferJobList(jobs=jobs)


@router.get("/{job_id}", response_model=InferJobState)
async def get_inference_job(job_id: str, request: Request) -> InferJobState:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    manager.reap_finished()
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "inference job not found")
    # Refresh image list on read (output dir may have images not yet finalized).
    state["images"] = manager._list_output_images(jid)
    return InferJobState(**_normalize_state(state))


@router.delete("/{job_id}", response_model=OkResponse)
async def delete_inference(job_id: str, request: Request) -> OkResponse:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    manager.reap_finished()
    if manager.is_alive(jid):
        raise HTTPException(status.HTTP_409_CONFLICT, "inference is still running; stop it first")
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "inference job not found")
    jd = manager.job_dir(jid)
    if jd.exists():
        shutil.rmtree(jd)
    return OkResponse()


@router.post("/{job_id}/stop", response_model=InferJobState)
async def stop_inference(job_id: str, request: Request) -> InferJobState:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if not manager.read_state(jid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "inference job not found")
    state = manager.stop(jid)
    return InferJobState(**_normalize_state(state))


@router.get("/{job_id}/events")
async def inference_events(job_id: str, request: Request) -> StreamingResponse:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    state = manager.read_state(jid)
    if not state:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "inference job not found")

    event_log = manager.event_log_path(jid)

    async def event_stream():
        yield ": stream open\n\n"
        try:
            async for record in tail_jsonl(
                event_log,
                is_job_alive=lambda: manager.is_alive(jid),
            ):
                if record.get("type") == "__heartbeat__":
                    yield ": heartbeat\n\n"
                    continue
                yield sse_frame(record, event=record.get("type"))
                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield sse_frame({"type": "error", "error": str(e)}, event="error")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.get("/{job_id}/images/{filename}")
async def serve_image(job_id: str, filename: str, request: Request) -> FileResponse:
    manager = _manager(request)
    try:
        jid = safe_component(job_id)
        fname = safe_filename(filename)
    except UnsafePathError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    path = resolve_under(manager.output_dir(jid), fname)
    if not path.exists() or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "image not found")
    return FileResponse(path, media_type="image/png")

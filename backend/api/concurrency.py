"""GPU concurrency guard.

Single 8GB consumer GPU assumption: only one training OR one inference
subprocess may be live at a time. Two workers racing for VRAM will OOM
before either produces useful output, so we reject the second request
at the API layer with 409 Conflict.
"""

from __future__ import annotations

from typing import Optional, Tuple

from fastapi import HTTPException, Request, status


def active_gpu_job(request: Request) -> Optional[Tuple[str, str]]:
    """Return (kind, job_id) if any GPU-using worker is live, else None."""
    train_mgr = getattr(request.app.state, "job_manager", None)
    infer_mgr = getattr(request.app.state, "inference_manager", None)

    if train_mgr is not None:
        train_mgr.reap_finished()
        jid = train_mgr.has_running_job()
        if jid:
            return ("train", jid)

    if infer_mgr is not None:
        infer_mgr.reap_finished()
        jid = infer_mgr.has_running_job()
        if jid:
            return ("inference", jid)

    return None


def require_gpu_free(request: Request) -> None:
    """Raise 409 if another GPU worker is already running."""
    active = active_gpu_job(request)
    if active is None:
        return
    kind, jid = active
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"GPU busy: {kind} job {jid} is running. Stop it before starting another.",
    )

"""Model (trained weights) listing.

MVP: a "model" is the `output/` directory of a completed job. We detect
LoRA vs. full by probing for `pytorch_lora_weights.bin` vs. `model_index.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter

from .. import settings
from ..schemas import ModelInfo, ModelList
from ..state import read_json

router = APIRouter(prefix="/models", tags=["models"])


def _probe(dir_path: Path) -> str | None:
    if (dir_path / "pytorch_lora_weights.bin").exists():
        return "lora"
    if (dir_path / "model_index.json").exists():
        return "full"
    return None


@router.get("", response_model=ModelList)
async def list_models() -> ModelList:
    settings.ensure_dirs()
    out: List[ModelInfo] = []
    for jd in sorted(settings.JOBS_DIR.iterdir()) if settings.JOBS_DIR.exists() else []:
        if not jd.is_dir():
            continue
        state = read_json(jd / "state.json")
        if not state or state.get("status") != "completed":
            continue
        out_dir = jd / "output"
        kind = _probe(out_dir)
        if kind is None:
            continue
        out.append(ModelInfo(
            id=state["id"],
            job_id=state["id"],
            path=str(out_dir),
            kind=kind,
            created_at=state.get("finished_at") or state.get("created_at", 0),
        ))
    out.sort(key=lambda m: m.created_at, reverse=True)
    return ModelList(models=out)


def resolve_model(model_id: str) -> ModelInfo | None:
    """Look up a model by id. Returns None if not found or incomplete."""
    from ..paths import UnsafePathError, safe_component
    try:
        mid = safe_component(model_id)
    except UnsafePathError:
        return None
    jd = settings.JOBS_DIR / mid
    if not jd.is_dir():
        return None
    state = read_json(jd / "state.json")
    if not state or state.get("status") != "completed":
        return None
    out_dir = jd / "output"
    kind = _probe(out_dir)
    if kind is None:
        return None
    return ModelInfo(
        id=state["id"],
        job_id=state["id"],
        path=str(out_dir),
        kind=kind,
        created_at=state.get("finished_at") or state.get("created_at", 0),
    )

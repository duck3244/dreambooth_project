"""Subprocess entry point for training.

Usage:
    python -m backend.api.workers.train_worker \\
        --config /path/to/config.json \\
        --event-log /path/to/events.jsonl

The config JSON is a dict matching `DreamBoothConfig` fields.
All training progress is appended to `--event-log` as JSONL records
consumable by the API layer for SSE streaming.
"""

import sys

# Guard against a broken xformers build (e.g. py310_pt env has xformers
# compiled for a newer torch than what's installed, raising
# `torch.backends.cuda has no attribute is_flash_attention_available`).
# Setting these to None BEFORE diffusers imports forces diffusers to fall
# back to SDPA. Opt out by setting DREAMBOOTH_ENABLE_XFORMERS=1.
import os as _os
if _os.environ.get("DREAMBOOTH_ENABLE_XFORMERS", "0") != "1":
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None

import argparse
import json
import os
import traceback
from pathlib import Path

# Make `backend.*` importable whether run as `python -m backend.api.workers.train_worker`
# (project root on PYTHONPATH) or as `python backend/api/workers/train_worker.py`.
_THIS = Path(__file__).resolve()
_BACKEND_DIR = _THIS.parents[2]
_PROJECT_ROOT = _BACKEND_DIR.parent
for p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config(path: str):
    """Load config dict from JSON and hydrate into a DreamBoothConfig instance."""
    from config import DreamBoothConfig

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_fields = {f for f in DreamBoothConfig.__dataclass_fields__.keys()}
    unknown = set(data.keys()) - valid_fields
    if unknown:
        # Not fatal — log and drop. The API layer may include extra metadata.
        print(f"[train_worker] ignoring unknown config fields: {unknown}", file=sys.stderr)
        data = {k: v for k, v in data.items() if k in valid_fields}

    return DreamBoothConfig(**data)


def _append_error_event(event_log: str, error: str, error_type: str) -> None:
    """Best-effort: write an error record to the event log if the trainer
    failed before reaching its own try/finally block (e.g., config load error)."""
    try:
        import time
        Path(event_log).parent.mkdir(parents=True, exist_ok=True)
        with open(event_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "type": "error",
                "step": 0,
                "error": error,
                "error_type": error_type,
            }) + "\n")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="DreamBooth training subprocess worker")
    ap.add_argument("--config", required=True, help="Path to config JSON file")
    ap.add_argument("--event-log", dest="event_log", required=True,
                    help="Path to append JSONL events (created if missing)")
    ap.add_argument("--resume-from", dest="resume_from", default=None,
                    help="Optional checkpoint path to resume from")
    args = ap.parse_args()

    # Import lazily so a bad config fails cheaply before loading torch/diffusers.
    try:
        config = _load_config(args.config)
    except Exception as e:
        _append_error_event(args.event_log, str(e), type(e).__name__)
        traceback.print_exc()
        return 2

    try:
        from trainer import DreamBoothTrainer
    except Exception as e:
        _append_error_event(args.event_log, str(e), type(e).__name__)
        traceback.print_exc()
        return 3

    trainer = DreamBoothTrainer(config, event_log_path=args.event_log)
    try:
        if args.resume_from:
            trainer.resume_from_checkpoint(args.resume_from)
        trainer.train()
    except KeyboardInterrupt:
        # Already emitted via trainer.train's except clause in most cases,
        # but add a best-effort notice to stderr.
        print("[train_worker] interrupted", file=sys.stderr)
        return 130
    except Exception:
        # trainer.train itself emits an 'error' event and re-raises.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

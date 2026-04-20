"""Atomic JSON state storage.

All mutable state files (job state.json, dataset metadata) are written via
tempfile + os.replace to survive crashes mid-write. Readers are lock-free
and simply read the file; torn reads are impossible because os.replace is
atomic on the same filesystem.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    """Return parsed JSON or `default` when the file does not exist."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write `data` as JSON to `path`.

    Uses NamedTemporaryFile in the same directory (so os.replace is atomic
    on the same filesystem) and fsyncs the file before rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

"""Path safety utilities.

All user-supplied identifiers (dataset_id, job_id, filename) flow through
`safe_component()` and `resolve_under()` before touching the filesystem.
This prevents traversal (`../../etc/passwd`), absolute-path escape, and
embedded null bytes.
"""

from __future__ import annotations

import re
from pathlib import Path

_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class UnsafePathError(ValueError):
    """User-supplied path fails the safety check."""


def safe_component(name: str) -> str:
    """Validate a single path component (no slashes, no dots-only, no leading dash).

    Allowed: [A-Za-z0-9][A-Za-z0-9._-]{0,127}
    """
    if not isinstance(name, str) or not name:
        raise UnsafePathError("component must be a non-empty string")
    if "\x00" in name or "/" in name or "\\" in name:
        raise UnsafePathError(f"component contains invalid chars: {name!r}")
    if name in (".", "..") or name.startswith("."):
        raise UnsafePathError(f"component cannot start with dot: {name!r}")
    if not _COMPONENT_RE.match(name):
        raise UnsafePathError(f"component does not match allowed pattern: {name!r}")
    return name


def resolve_under(base: Path, *parts: str) -> Path:
    """Resolve `parts` under `base`, asserting the result stays inside `base`.

    Accepts multi-segment relative inputs (e.g. a user upload filename). Each
    segment is validated. Symlinks are resolved before the containment check.
    """
    base_resolved = base.resolve()
    candidate = base_resolved
    for part in parts:
        # Split compound inputs (defense in depth: reject any slash first)
        if "/" in part or "\\" in part:
            raise UnsafePathError(f"path part must be a single component: {part!r}")
        safe_component(part)
        candidate = candidate / part

    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise UnsafePathError(f"path escapes base: {resolved} not under {base_resolved}") from exc
    return resolved


def safe_filename(name: str) -> str:
    """Validate a user-uploaded filename. Keeps the original extension case."""
    if not name or "\x00" in name:
        raise UnsafePathError("invalid filename")
    base = Path(name).name  # strip any directories the client may have sent
    if not base or base in (".", ".."):
        raise UnsafePathError(f"invalid filename: {name!r}")
    # Must have exactly one extension segment, no hidden files.
    if base.startswith("."):
        raise UnsafePathError(f"hidden filename not allowed: {name!r}")
    stem = Path(base).stem
    if not stem or not _COMPONENT_RE.match(stem):
        raise UnsafePathError(f"filename stem is invalid: {name!r}")
    return base

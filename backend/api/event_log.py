"""JSONL tail → SSE stream.

Each SSE connection runs an independent async tailer: maintain a byte
offset, reread the file whenever new bytes appear, parse each complete
line as JSON, and yield an SSE frame. The tailer stops when a terminal
event (`completed` / `error`) is observed, so the HTTP stream closes
cleanly — the client's EventSource will not reconnect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

_TERMINAL_TYPES = {"completed", "error", "cancelled"}


def sse_frame(data: dict, event: Optional[str] = None) -> str:
    """Format one SSE frame. The client decodes `data` as JSON."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    # JSON payload may contain newlines — SSE requires splitting on \n.
    payload = json.dumps(data, ensure_ascii=False)
    for p in payload.splitlines() or [""]:
        lines.append(f"data: {p}")
    lines.append("")  # trailing blank = frame boundary
    lines.append("")
    return "\n".join(lines)


async def tail_jsonl(
    path: Path,
    *,
    poll_interval: float = 0.25,
    initial_wait: float = 5.0,
    heartbeat_interval: float = 15.0,
    is_job_alive: Optional[callable] = None,
    terminal_types: Iterable[str] = _TERMINAL_TYPES,
) -> AsyncIterator[dict]:
    """Async generator yielding parsed JSON records from `path` as they arrive.

    - Waits up to `initial_wait` seconds for the file to appear (subprocess
      may race with the HTTP request).
    - Emits a `__heartbeat__` record every `heartbeat_interval` seconds when
      nothing new has arrived; the caller can translate this to an SSE
      comment frame to keep proxies from timing out.
    - Terminates after yielding any record whose `type` is in
      `terminal_types`, or when `is_job_alive()` returns False and no bytes
      remain to read.
    """
    terminal_set = set(terminal_types)

    # Wait for file to exist
    waited = 0.0
    while not path.exists():
        if waited >= initial_wait:
            yield {"type": "error", "error": f"event log not found: {path.name}"}
            return
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    offset = 0
    buffer = b""
    last_activity = asyncio.get_event_loop().time()

    while True:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            # Log rotated/removed under our feet
            return

        if size > offset:
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset)
            except FileNotFoundError:
                return
            buffer += chunk
            offset = size

            # Split on newline; keep the trailing partial line in buffer.
            *complete, buffer = buffer.split(b"\n")
            for raw in complete:
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    yield {"type": "parse_error", "raw": raw.decode("utf-8", "replace"), "error": str(e)}
                    continue
                yield record
                last_activity = asyncio.get_event_loop().time()
                if record.get("type") in terminal_set:
                    return
        else:
            # Nothing new; check liveness and emit heartbeat if idle.
            now = asyncio.get_event_loop().time()
            if now - last_activity >= heartbeat_interval:
                yield {"type": "__heartbeat__"}
                last_activity = now

            if is_job_alive is not None and not is_job_alive():
                # Allow one final drain pass
                try:
                    final_size = path.stat().st_size
                except FileNotFoundError:
                    return
                if final_size <= offset:
                    # Subprocess is gone and there's nothing left to read.
                    yield {"type": "cancelled", "reason": "subprocess exited without terminal event"}
                    return

            await asyncio.sleep(poll_interval)

"""pynvml-based GPU inspection for the API process.

Why pynvml (not torch): importing torch in the API server initializes CUDA
runtime, loads cuDNN/BLAS, and holds ~1GB of VRAM before any training has
started — starving the training subprocess. pynvml speaks to NVML directly
and does not touch CUDA context.
"""

from __future__ import annotations

import threading
from typing import Optional

from .schemas import GPUInfo


_lock = threading.Lock()
_initialized = False


def init() -> None:
    """Idempotently initialize NVML. Safe to call multiple times."""
    global _initialized
    with _lock:
        if _initialized:
            return
        try:
            import pynvml  # lazy import — fine since API already depends on it
            pynvml.nvmlInit()
            _initialized = True
        except Exception:
            # If NVML is unavailable (no driver, no GPU), we record that the
            # GPU is unavailable but do not crash the API.
            _initialized = False


def shutdown() -> None:
    global _initialized
    with _lock:
        if not _initialized:
            return
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _initialized = False


def _decode(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="replace")
    return str(value)


def get_info(index: int = 0) -> GPUInfo:
    """Snapshot of GPU `index`. Never raises."""
    if not _initialized:
        init()
    if not _initialized:
        return GPUInfo(available=False, error="NVML not available")

    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = _decode(pynvml.nvmlDeviceGetName(handle))
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None
        try:
            driver = _decode(pynvml.nvmlSystemGetDriverVersion())
        except Exception:
            driver = None
        return GPUInfo(
            available=True,
            name=name,
            driver_version=driver,
            memory_total_mb=int(mem.total // (1024 * 1024)),
            memory_used_mb=int(mem.used // (1024 * 1024)),
            memory_free_mb=int(mem.free // (1024 * 1024)),
            utilization_percent=int(util.gpu),
            temperature_c=temp,
        )
    except Exception as e:
        return GPUInfo(available=False, error=str(e))

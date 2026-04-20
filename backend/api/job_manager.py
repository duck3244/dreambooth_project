"""Training job lifecycle.

Responsibilities:
- Persist each job as `<JOBS_DIR>/<job_id>/state.json` (atomic writes).
- Spawn the training worker as a subprocess in a new process group so
  SIGTERM to the leader reliably reaps the child tree.
- Poll-based reaping: `reap_finished()` is cheap and called from request
  handlers and the lifespan background task.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import settings
from .paths import resolve_under, safe_component
from .state import read_json, write_json_atomic


STATE_FILENAME = "state.json"
CONFIG_FILENAME = "config.json"
EVENT_LOG_FILENAME = "events.jsonl"
STDOUT_LOG_FILENAME = "worker.log"


@dataclass
class _RunningProc:
    popen: subprocess.Popen
    stdout_fp: object


class JobManager:
    def __init__(self, jobs_dir: Optional[Path] = None):
        self.jobs_dir: Path = Path(jobs_dir or settings.JOBS_DIR)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._procs: Dict[str, _RunningProc] = {}
        self._lock = threading.Lock()

    # ---------- paths ----------

    def job_dir(self, job_id: str) -> Path:
        return resolve_under(self.jobs_dir, safe_component(job_id))

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / STATE_FILENAME

    def event_log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / EVENT_LOG_FILENAME

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    # ---------- state I/O ----------

    def read_state(self, job_id: str) -> Optional[dict]:
        return read_json(self.state_path(job_id))

    def write_state(self, job_id: str, state: dict) -> None:
        write_json_atomic(self.state_path(job_id), state)

    def update_state(self, job_id: str, **fields) -> dict:
        current = self.read_state(job_id) or {}
        current.update(fields)
        self.write_state(job_id, current)
        return current

    # ---------- lifecycle ----------

    def start(self, config_dict: dict, *, dataset_id: Optional[str] = None) -> dict:
        """Create a new job, spawn the worker, and return the initial state."""
        job_id = uuid.uuid4().hex[:12]
        jd = self.jobs_dir / job_id
        jd.mkdir(parents=True, exist_ok=True)

        output_dir = jd / "output"
        logs_dir = jd / "logs"
        output_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        # Training config: force output paths under the job dir to prevent
        # untrusted clients from steering writes elsewhere.
        config_dict = dict(config_dict)
        config_dict["output_dir"] = str(output_dir)
        config_dict["logging_dir"] = str(logs_dir)

        config_path = jd / CONFIG_FILENAME
        write_json_atomic(config_path, config_dict)

        event_log_path = jd / EVENT_LOG_FILENAME
        # Pre-create to avoid SSE `initial_wait` race.
        event_log_path.touch()

        stdout_path = jd / STDOUT_LOG_FILENAME
        stdout_fp = open(stdout_path, "ab", buffering=0)

        env = os.environ.copy()
        # Ensure project root is first on PYTHONPATH so `backend.*` imports resolve.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(settings.PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        # Keep TF32 / allocator env from config helper for parity with CLI.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        cmd = [
            sys.executable, "-m", settings.TRAIN_WORKER_MODULE,
            "--config", str(config_path),
            "--event-log", str(event_log_path),
        ]

        # start_new_session=True puts the child in its own process group so
        # SIGTERM on stop() reliably reaches any grandchildren too.
        popen = subprocess.Popen(
            cmd,
            stdout=stdout_fp,
            stderr=subprocess.STDOUT,
            cwd=str(settings.BACKEND_DIR),
            env=env,
            start_new_session=True,
        )

        now = time.time()
        state = {
            "id": job_id,
            "status": "running",
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "pid": popen.pid,
            "return_code": None,
            "error": None,
            "dataset_id": dataset_id,
            "output_dir": str(output_dir),
            "event_log": str(event_log_path),
            "max_train_steps": config_dict.get("max_train_steps"),
            "use_lora": config_dict.get("use_lora"),
            "latest_step": 0,
            "latest_loss": None,
        }
        self.write_state(job_id, state)
        with self._lock:
            self._procs[job_id] = _RunningProc(popen=popen, stdout_fp=stdout_fp)
        return state

    def is_alive(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is None:
            return False
        return proc.popen.poll() is None

    def has_running_job(self) -> Optional[str]:
        """Return a job_id if any worker is still live, else None.
        Used as a VRAM-concurrency guard on an 8GB GPU."""
        with self._lock:
            items = list(self._procs.items())
        for jid, proc in items:
            if proc.popen.poll() is None:
                return jid
        return None

    def stop(self, job_id: str, grace: float = 5.0) -> dict:
        """Request graceful shutdown; SIGKILL after `grace` seconds.

        Sends signals to the full process group so any grandchildren (DDP
        workers, dataloaders) also die.
        """
        with self._lock:
            proc = self._procs.get(job_id)

        state = self.read_state(job_id) or {"id": job_id}
        if proc is None or proc.popen.poll() is not None:
            if state.get("status") == "running":
                state["status"] = "failed"
                state["error"] = state.get("error") or "process already gone"
                self.write_state(job_id, state)
            return state

        pgid = os.getpgid(proc.popen.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.time() + grace
        while time.time() < deadline:
            if proc.popen.poll() is not None:
                break
            time.sleep(0.1)

        if proc.popen.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.popen.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        self._finalize(job_id, cancelled=True)
        return self.read_state(job_id) or {}

    def reap_finished(self) -> None:
        """Scan running procs and update state for any that exited.

        Skip jobs whose state.json already reflects a terminal status — this
        happens when stop() has already finalized the job as `cancelled`; the
        subprocess exits shortly after SIGTERM and we don't want the reaper
        to reclassify it as `failed`.
        """
        done: List[str] = []
        with self._lock:
            items = list(self._procs.items())
        for jid, proc in items:
            if proc.popen.poll() is not None:
                state = self.read_state(jid) or {}
                if state.get("status") in {"completed", "failed", "cancelled"}:
                    # Already finalized elsewhere — just release resources.
                    try:
                        proc.stdout_fp.close()
                    except Exception:
                        pass
                else:
                    self._finalize(jid, cancelled=False)
                done.append(jid)
        with self._lock:
            for jid in done:
                self._procs.pop(jid, None)

    def _finalize(self, job_id: str, *, cancelled: bool) -> None:
        with self._lock:
            proc = self._procs.get(job_id)
        rc = None
        if proc is not None:
            rc = proc.popen.returncode
            try:
                proc.stdout_fp.close()
            except Exception:
                pass

        state = self.read_state(job_id) or {"id": job_id}
        state["finished_at"] = time.time()
        state["return_code"] = rc
        if cancelled:
            state["status"] = "cancelled"
        else:
            # Trust the event log for success/error signal — return code alone
            # can be zero even when training emitted an error event, or nonzero
            # for SIGTERM on Windows-like platforms.
            state["status"] = "completed" if rc == 0 else "failed"
            if rc is not None and rc != 0 and not state.get("error"):
                state["error"] = f"worker exited with code {rc}"
        self.write_state(job_id, state)

    # ---------- listing ----------

    def list_jobs(self) -> List[dict]:
        self.reap_finished()
        out: List[dict] = []
        if not self.jobs_dir.exists():
            return out
        for jd in sorted(self.jobs_dir.iterdir()):
            if not jd.is_dir():
                continue
            state = read_json(jd / STATE_FILENAME)
            if state:
                out.append(state)
        out.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return out

    # ---------- recovery ----------

    def reconcile_on_startup(self) -> None:
        """On server start, mark any job left in `running` status as failed.

        The pid from last run is meaningless after a restart, and we have no
        way to re-attach to an orphan subprocess reliably.
        """
        for jd in self.jobs_dir.iterdir() if self.jobs_dir.exists() else []:
            if not jd.is_dir():
                continue
            state = read_json(jd / STATE_FILENAME)
            if not state or state.get("status") != "running":
                continue
            # If the pid is still live AND belongs to a train_worker, we could
            # try to re-adopt — but MVP: treat as orphaned.
            state["status"] = "failed"
            state["error"] = state.get("error") or "server restarted while job was running"
            state["finished_at"] = state.get("finished_at") or time.time()
            write_json_atomic(jd / STATE_FILENAME, state)

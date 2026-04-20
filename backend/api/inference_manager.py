"""Inference job lifecycle.

Each inference request becomes a short-lived subprocess. One-shot, no
long-running state — but we reuse the same JSONL event log + subprocess
pattern as training so the SSE tail code works unchanged.

Layout:  INFERENCE_DIR/<infer_id>/
            state.json        - job state (matches InferJobState)
            request.json      - worker input
            events.jsonl      - JSONL events (started, image, completed, error)
            output/           - generated images (served back via the API)
            worker.log        - stdout/stderr of the worker process
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
REQUEST_FILENAME = "request.json"
EVENT_LOG_FILENAME = "events.jsonl"
STDOUT_LOG_FILENAME = "worker.log"
OUTPUT_DIRNAME = "output"


@dataclass
class _RunningProc:
    popen: subprocess.Popen
    stdout_fp: object


class InferenceManager:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir: Path = Path(base_dir or settings.INFERENCE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._procs: Dict[str, _RunningProc] = {}
        self._lock = threading.Lock()

    # ---------- paths ----------

    def job_dir(self, job_id: str) -> Path:
        return resolve_under(self.base_dir, safe_component(job_id))

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / STATE_FILENAME

    def event_log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / EVENT_LOG_FILENAME

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / OUTPUT_DIRNAME

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

    def start(
        self,
        *,
        model_id: str,
        model_path: str,
        model_kind: str,
        base_model: str,
        prompts: List[str],
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        num_images_per_prompt: int,
        seed: Optional[int],
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        jd = self.base_dir / job_id
        jd.mkdir(parents=True, exist_ok=True)
        out_dir = jd / OUTPUT_DIRNAME
        out_dir.mkdir(exist_ok=True)

        request_dict = {
            "model_id": model_id,
            "model_path": model_path,
            "model_kind": model_kind,
            "base_model": base_model,
            "prompts": prompts,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_images_per_prompt": num_images_per_prompt,
            "seed": seed,
        }
        request_path = jd / REQUEST_FILENAME
        write_json_atomic(request_path, request_dict)

        event_log_path = jd / EVENT_LOG_FILENAME
        event_log_path.touch()

        stdout_path = jd / STDOUT_LOG_FILENAME
        stdout_fp = open(stdout_path, "ab", buffering=0)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(settings.PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        cmd = [
            sys.executable, "-m", settings.INFER_WORKER_MODULE,
            "--request", str(request_path),
            "--event-log", str(event_log_path),
            "--output-dir", str(out_dir),
        ]

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
            "model_id": model_id,
            "created_at": now,
            "started_at": now,
            "finished_at": None,
            "pid": popen.pid,
            "return_code": None,
            "error": None,
            "output_dir": str(out_dir),
            "event_log": str(event_log_path),
            "prompts": prompts,
            "images": [],
            "total_images": len(prompts) * num_images_per_prompt,
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
        with self._lock:
            items = list(self._procs.items())
        for jid, proc in items:
            if proc.popen.poll() is None:
                return jid
        return None

    def stop(self, job_id: str, grace: float = 3.0) -> dict:
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
        done: List[str] = []
        with self._lock:
            items = list(self._procs.items())
        for jid, proc in items:
            if proc.popen.poll() is not None:
                state = self.read_state(jid) or {}
                if state.get("status") in {"completed", "failed", "cancelled"}:
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

        # Always refresh image list from disk — events may still be flushing.
        state["images"] = self._list_output_images(job_id)

        if cancelled:
            state["status"] = "cancelled"
        else:
            state["status"] = "completed" if rc == 0 else "failed"
            if rc is not None and rc != 0 and not state.get("error"):
                state["error"] = f"worker exited with code {rc}"
        self.write_state(job_id, state)

    def _list_output_images(self, job_id: str) -> List[str]:
        out = self.output_dir(job_id)
        if not out.exists():
            return []
        return sorted(p.name for p in out.iterdir()
                      if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))

    # ---------- listing ----------

    def list_jobs(self) -> List[dict]:
        self.reap_finished()
        out: List[dict] = []
        if not self.base_dir.exists():
            return out
        for jd in sorted(self.base_dir.iterdir()):
            if not jd.is_dir():
                continue
            state = read_json(jd / STATE_FILENAME)
            if state:
                out.append(state)
        out.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return out

    # ---------- recovery ----------

    def reconcile_on_startup(self) -> None:
        for jd in self.base_dir.iterdir() if self.base_dir.exists() else []:
            if not jd.is_dir():
                continue
            state = read_json(jd / STATE_FILENAME)
            if not state or state.get("status") != "running":
                continue
            state["status"] = "failed"
            state["error"] = state.get("error") or "server restarted while inference was running"
            state["finished_at"] = state.get("finished_at") or time.time()
            write_json_atomic(jd / STATE_FILENAME, state)

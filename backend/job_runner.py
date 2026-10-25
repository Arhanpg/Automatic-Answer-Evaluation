"""
backend/job_runner.py -- Stateful Background Job Worker & Resumable SSE Streamer.

Decouples long-running GPU OCR and ML/DL/LLM grading from synchronous HTTP requests.
Persists every step to SQLite (WAL mode) so switching browser tabs or refreshing never loses state.
"""
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, Generator, Optional

from backend.database import (
    create_background_job,
    get_active_background_job,
    get_background_job,
    update_background_job,
)
from backend.gpu_manager import gpu_manager

logger = logging.getLogger("job_runner")


class BackgroundJobRunner:
    """
    Singleton Background Worker managing asynchronous GPU jobs with SQLite checkpointing
    and Server-Sent Events (SSE) streaming.
    """
    _instance: Optional["BackgroundJobRunner"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BackgroundJobRunner, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_job_id: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._job_events: Dict[str, queue.Queue] = {}
        self._events_lock = threading.Lock()

    def get_event_queue(self, job_id: str) -> queue.Queue:
        with self._events_lock:
            if job_id not in self._job_events:
                self._job_events[job_id] = queue.Queue(maxsize=100)
            return self._job_events[job_id]

    def emit_event(self, job_id: str, event_data: dict):
        q = self.get_event_queue(job_id)
        try:
            q.put_nowait(event_data)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(event_data)
            except Exception:
                pass

    def start_job(
        self,
        job_type: str,
        worker_fn: Callable[["JobContext"], Any],
        total_steps: int = 1,
    ) -> str:
        """
        Starts a background job. If an active job is already running, returns its job_id.
        Otherwise creates a new job in SQLite and spawns a worker thread.
        """
        active = get_active_background_job(job_type)
        if active and active.get("status") == "running":
            logger.info("Found already running job %s of type %s", active["job_id"], job_type)
            return active["job_id"]

        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self._current_job_id = job_id
        create_background_job(job_id, job_type, total_steps)

        def runner_target():
            ctx = JobContext(job_id, job_type, total_steps, self)
            try:
                logger.info("Starting background job %s (%s)", job_id, job_type)
                result = worker_fn(ctx)
                ctx.finish(result=result)
            except Exception as e:
                logger.exception("Error in background job %s: %s", job_id, e)
                ctx.fail(str(e))
            finally:
                if self._current_job_id == job_id:
                    self._current_job_id = None

        self._worker_thread = threading.Thread(target=runner_target, daemon=True)
        self._worker_thread.start()
        return job_id

    def stream_job(self, job_id: str) -> Generator[str, None, None]:
        """
        Server-Sent Events generator streaming job updates.
        Immediately yields the current checkpoint from SQLite, then streams live events.
        """
        # Initial snapshot from SQLite
        job = get_background_job(job_id)
        if job:
            vram = gpu_manager.get_vram_telemetry()
            job["vram_telemetry"] = vram
            yield f"data: {json.dumps(job)}\n\n"
            if job.get("status") in ("completed", "failed"):
                return

        event_q = self.get_event_queue(job_id)
        while True:
            try:
                event = event_q.get(timeout=0.8)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("completed", "failed"):
                    break
            except queue.Empty:
                # Periodic heartbeat check against SQLite
                current = get_background_job(job_id)
                if not current:
                    break
                vram = gpu_manager.get_vram_telemetry()
                current["vram_telemetry"] = vram
                yield f"data: {json.dumps(current)}\n\n"
                if current.get("status") in ("completed", "failed"):
                    break


class JobContext:
    """Provides checkpointing, logging, and progress updates to a running job."""

    def __init__(self, job_id: str, job_type: str, total_steps: int, runner: BackgroundJobRunner):
        self.job_id = job_id
        self.job_type = job_type
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.runner = runner
        self.logs = []
        self.result_summary = {}

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logger.info("[Job %s] %s", self.job_id, message)

    def checkpoint(
        self,
        completed_steps: Optional[int] = None,
        current_item: str = "",
        current_stage: str = "",
        partial_result: Optional[dict] = None,
    ):
        if completed_steps is not None:
            self.completed_steps = completed_steps

        progress_pct = round((self.completed_steps / self.total_steps) * 100.0, 1)
        progress_pct = min(max(progress_pct, 0.0), 100.0)

        vram = gpu_manager.get_vram_telemetry()
        vram_mb = vram.get("allocated_mb", 0.0)

        if partial_result:
            self.result_summary.update(partial_result)

        update_background_job(
            self.job_id,
            status="running",
            progress_pct=progress_pct,
            completed_steps=self.completed_steps,
            current_item=current_item,
            current_stage=current_stage,
            vram_mb=vram_mb,
            logs_json=self.logs,
            result_summary_json=self.result_summary,
        )

        event_payload = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": "running",
            "progress_pct": progress_pct,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "current_item": current_item,
            "current_stage": current_stage,
            "vram_mb": vram_mb,
            "vram_telemetry": vram,
            "logs": self.logs,
            "result_summary": self.result_summary,
        }
        self.runner.emit_event(self.job_id, event_payload)

    def finish(self, result: Optional[dict] = None):
        vram = gpu_manager.get_vram_telemetry()
        vram_mb = vram.get("allocated_mb", 0.0)
        if result:
            self.result_summary.update(result)

        update_background_job(
            self.job_id,
            status="completed",
            progress_pct=100.0,
            completed_steps=self.total_steps,
            current_stage="done",
            vram_mb=vram_mb,
            logs_json=self.logs,
            result_summary_json=self.result_summary,
        )

        event_payload = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": "completed",
            "progress_pct": 100.0,
            "total_steps": self.total_steps,
            "completed_steps": self.total_steps,
            "current_stage": "done",
            "vram_mb": vram_mb,
            "vram_telemetry": vram,
            "logs": self.logs,
            "result_summary": self.result_summary,
        }
        self.runner.emit_event(self.job_id, event_payload)

    def fail(self, error_text: str):
        self.log(f"FAILED: {error_text}")
        vram = gpu_manager.get_vram_telemetry()
        vram_mb = vram.get("allocated_mb", 0.0)

        update_background_job(
            self.job_id,
            status="failed",
            error_text=error_text,
            current_stage="error",
            vram_mb=vram_mb,
            logs_json=self.logs,
            result_summary_json=self.result_summary,
        )

        event_payload = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": "failed",
            "error_text": error_text,
            "current_stage": "error",
            "vram_mb": vram_mb,
            "vram_telemetry": vram,
            "logs": self.logs,
            "result_summary": self.result_summary,
        }
        self.runner.emit_event(self.job_id, event_payload)


# Global singleton instance
job_runner = BackgroundJobRunner()

"""
非同期ジョブ管理（in-memory + ThreadPoolExecutor）。

ラボ規模運用向け。uvicorn worker=1 を前提とする。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.simulation import SimulationResult, execute_simulation
from app.solver_runner import terminate_solver

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_input_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class JobRecord:
    run_id: str
    status: str  # queued|running|succeeded|failed|cancelled
    input_hash: str
    created_at: str
    request: Dict[str, Any]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress_stage: str = "queued"
    progress_message: str = ""
    artifact_dir: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # SimulationResponse-compatible dict
    error: Optional[Dict[str, Any]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Optional[Future] = field(default=None, repr=False)

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "input_hash": self.input_hash,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "stage": self.progress_stage,
                "message": self.progress_message,
            },
            "artifact_dir": self.artifact_dir,
            "error": self.error,
        }


class RunManager:
    def __init__(self, *, max_workers: Optional[int] = None) -> None:
        if max_workers is None:
            try:
                max_workers = int(os.getenv("VTSIMNX_MAX_WORKERS", "1"))
            except ValueError:
                max_workers = 1
        max_workers = max(1, int(max_workers))
        self._max_workers = max_workers
        self._lock = threading.RLock()
        self._jobs: Dict[str, JobRecord] = {}
        self._hash_index: Dict[str, str] = {}  # input_hash -> run_id for queued/running
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vtsimnx-run")

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def submit(self, request: Dict[str, Any]) -> tuple[JobRecord, bool]:
        """
        request: SimulationRequest 相当の dict（config/debug/...）
        Returns: (job, is_duplicate)
        """
        input_hash = compute_input_hash(request)
        with self._lock:
            existing_id = self._hash_index.get(input_hash)
            if existing_id:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.status in ("queued", "running"):
                    return existing, True

            run_id = uuid.uuid4().hex
            job = JobRecord(
                run_id=run_id,
                status="queued",
                input_hash=input_hash,
                created_at=_utc_now(),
                request=request,
            )
            self._jobs[run_id] = job
            self._hash_index[input_hash] = run_id
            job.future = self._executor.submit(self._run_job, run_id)
            return job, False

    def get(self, run_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(run_id)

    def cancel(self, run_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            if job.status in ("succeeded", "failed", "cancelled"):
                return job
            job.cancel_event.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = _utc_now()
                job.progress_stage = "cancelled"
                job.progress_message = "cancelled before start"
                self._hash_index.pop(job.input_hash, None)
                if job.future is not None:
                    job.future.cancel()
            elif job.status == "running":
                terminate_solver(run_id)
            return job

    def _set_progress(self, run_id: str, stage: str, message: str = "") -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            job.progress_stage = stage
            job.progress_message = message

    def _run_job(self, run_id: str) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return
            if job.cancel_event.is_set() or job.status == "cancelled":
                job.status = "cancelled"
                job.finished_at = _utc_now()
                self._hash_index.pop(job.input_hash, None)
                return
            job.status = "running"
            job.started_at = _utc_now()
            job.progress_stage = "running"
            req = dict(job.request)
            cancel_event = job.cancel_event

        try:
            result = execute_simulation(
                req.get("config") or {},
                run_id=run_id,
                debug=bool(req.get("debug", False)),
                debug_verbosity=int(req.get("debug_verbosity", 2)),
                add_surface=req.get("add_surface"),
                add_aircon=req.get("add_aircon"),
                add_capacity=req.get("add_capacity"),
                add_moisture_capacity=req.get("add_moisture_capacity"),
                add_surface_solar=req.get("add_surface_solar"),
                add_surface_nocturnal=req.get("add_surface_nocturnal"),
                add_surface_radiation=req.get("add_surface_radiation"),
                add_surface_radiation_exclude_glass=req.get("add_surface_radiation_exclude_glass"),
                progress_cb=lambda stage, msg: self._set_progress(run_id, stage, msg),
                cancel_event=cancel_event,
                unknown_keys=str(req.get("unknown_keys") or "strip"),
                initial_warnings=req.get("initial_warnings") or [],
                initial_warning_details=req.get("initial_warning_details") or [],
            )
            with self._lock:
                job = self._jobs.get(run_id)
                if job is None:
                    return
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.finished_at = _utc_now()
                    job.progress_stage = "cancelled"
                    job.error = {"code": "cancelled", "message": "run cancelled"}
                else:
                    job.status = "succeeded"
                    job.finished_at = _utc_now()
                    job.progress_stage = "done"
                    job.artifact_dir = (
                        result.output.get("artifact_dir")
                        if isinstance(result.output.get("artifact_dir"), str)
                        else None
                    )
                    job.result = {
                        "result": result.output,
                        "warnings": result.warnings,
                        "warning_details": result.warning_details,
                    }
                self._hash_index.pop(job.input_hash, None)
        except Exception as e:
            from app.schemas import UnknownFieldError

            logger.exception("run job failed: %s", run_id)
            with self._lock:
                job = self._jobs.get(run_id)
                if job is None:
                    return
                if job.cancel_event.is_set() or "cancelled" in str(e).lower():
                    job.status = "cancelled"
                    job.error = {"code": "cancelled", "message": str(e)}
                elif isinstance(e, UnknownFieldError):
                    job.status = "failed"
                    job.error = {
                        "code": "unknown_field",
                        "message": str(e),
                        "details": e.details,
                        "run_id": run_id,
                    }
                else:
                    job.status = "failed"
                    job.error = {"code": "internal_error", "message": str(e), "run_id": run_id}
                job.finished_at = _utc_now()
                job.progress_stage = job.status
                self._hash_index.pop(job.input_hash, None)


_MANAGER: Optional[RunManager] = None
_MANAGER_LOCK = threading.Lock()


def get_run_manager() -> RunManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = RunManager()
        return _MANAGER


def set_run_manager(manager: Optional[RunManager]) -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager


__all__ = [
    "JobRecord",
    "RunManager",
    "compute_input_hash",
    "get_run_manager",
    "set_run_manager",
]

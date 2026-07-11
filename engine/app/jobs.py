"""
非同期ジョブ管理（in-memory + ThreadPoolExecutor）。

ラボ規模運用向け。uvicorn worker=1 を前提とする。

環境変数:
  VTSIMNX_MAX_WORKERS       ThreadPool 上限（既定 1）
  VTSIMNX_JOB_TTL_SEC       完了ジョブの保持秒（既定 86400=24h、0 で TTL 無効）
  VTSIMNX_MAX_JOB_RECORDS   ジョブレコード上限（既定 1000、0 で件数制限無効）
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

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def compute_input_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def output_indicates_failure(output: Any) -> bool:
    """solver / postprocess が error を返したか。"""
    if not isinstance(output, dict):
        return False
    status = output.get("status")
    if isinstance(status, str) and status.strip().lower() == "error":
        return True
    err = output.get("error")
    if isinstance(err, str) and err.strip():
        return True
    if isinstance(err, dict) and err:
        return True
    return False


def failure_from_output(output: Dict[str, Any], *, run_id: str) -> Dict[str, Any]:
    """ジョブ API 用の error オブジェクトを組み立てる。"""
    raw_err = output.get("error")
    top_code = output.get("error_code")
    if isinstance(raw_err, dict):
        code = str(raw_err.get("code") or top_code or "solver_error")
        message = str(raw_err.get("message") or raw_err)
        err: Dict[str, Any] = {"code": code, "message": message, "run_id": run_id}
        for k, v in raw_err.items():
            if k not in err:
                err[k] = v
    elif isinstance(raw_err, str) and raw_err.strip():
        message = raw_err.strip()
        if isinstance(top_code, str) and top_code.strip():
            code = top_code.strip()
        else:
            code = "artifact_quota_exceeded" if "per-run limit" in message else "solver_error"
        err = {"code": code, "message": message, "run_id": run_id}
    else:
        if isinstance(top_code, str) and top_code.strip():
            err = {"code": top_code.strip(), "message": "solver returned status=error", "run_id": run_id}
        else:
            err = {"code": "solver_error", "message": "solver returned status=error", "run_id": run_id}
    art = output.get("artifact_dir")
    if isinstance(art, str) and art:
        err["artifact_dir"] = art
    return err


def _parse_iso_to_epoch(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        # handle trailing Z
        text = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


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
    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        job_ttl_sec: Optional[int] = None,
        max_job_records: Optional[int] = None,
    ) -> None:
        if max_workers is None:
            max_workers = _env_int("VTSIMNX_MAX_WORKERS", 1)
        max_workers = max(1, int(max_workers))
        self._max_workers = max_workers
        self._job_ttl_sec = (
            _env_int("VTSIMNX_JOB_TTL_SEC", 24 * 3600) if job_ttl_sec is None else int(job_ttl_sec)
        )
        self._max_job_records = (
            _env_int("VTSIMNX_MAX_JOB_RECORDS", 1000) if max_job_records is None else int(max_job_records)
        )
        self._lock = threading.RLock()
        self._jobs: Dict[str, JobRecord] = {}
        self._hash_index: Dict[str, str] = {}  # input_hash -> run_id for queued/running
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vtsimnx-run")

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def job_count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def prune_jobs(self, *, now: Optional[float] = None) -> int:
        """完了ジョブを TTL / 最大件数で削除する。戻り値は削除件数。"""
        with self._lock:
            return self._prune_jobs_unlocked(now=now)

    def _prune_jobs_unlocked(self, *, now: Optional[float] = None) -> int:
        now_ts = time.time() if now is None else now
        removed = 0
        ttl = self._job_ttl_sec
        max_records = self._max_job_records

        if ttl > 0:
            expired: List[str] = []
            for rid, job in self._jobs.items():
                if job.status not in _TERMINAL:
                    continue
                finished = _parse_iso_to_epoch(job.finished_at) or _parse_iso_to_epoch(job.created_at)
                if finished is None:
                    continue
                if now_ts - finished > ttl:
                    expired.append(rid)
            for rid in expired:
                self._drop_job_unlocked(rid)
                removed += 1

        if max_records > 0 and len(self._jobs) > max_records:
            # 完了済みを古い順に削除（queued/running は残す）
            finished_jobs: List[tuple[float, str]] = []
            for rid, job in self._jobs.items():
                if job.status not in _TERMINAL:
                    continue
                finished = _parse_iso_to_epoch(job.finished_at) or _parse_iso_to_epoch(job.created_at) or 0.0
                finished_jobs.append((finished, rid))
            finished_jobs.sort(key=lambda x: x[0])
            overflow = len(self._jobs) - max_records
            for _ts, rid in finished_jobs:
                if overflow <= 0:
                    break
                self._drop_job_unlocked(rid)
                removed += 1
                overflow -= 1

        if removed:
            logger.info(
                "job prune audit: removed=%s remaining=%s ttl_sec=%s max_records=%s",
                removed,
                len(self._jobs),
                ttl,
                max_records,
            )
        return removed

    def _drop_job_unlocked(self, run_id: str) -> None:
        job = self._jobs.pop(run_id, None)
        if job is None:
            return
        if self._hash_index.get(job.input_hash) == run_id:
            self._hash_index.pop(job.input_hash, None)
        # 参照を切って GC しやすくする
        job.request = {}
        job.result = None
        job.future = None

    def submit(self, request: Dict[str, Any]) -> tuple[JobRecord, bool]:
        """
        request: SimulationRequest 相当の dict（config/debug/...）
        Returns: (job, is_duplicate)
        """
        input_hash = compute_input_hash(request)
        with self._lock:
            self._prune_jobs_unlocked()
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
            if job.status in _TERMINAL:
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

    def _finalize_success_or_output_error(self, job: JobRecord, result: SimulationResult) -> None:
        output = result.output if isinstance(result.output, dict) else {}
        artifact_dir = output.get("artifact_dir") if isinstance(output.get("artifact_dir"), str) else None
        job.artifact_dir = artifact_dir
        payload = {
            "result": output,
            "warnings": result.warnings,
            "warning_details": result.warning_details,
        }
        if output_indicates_failure(output):
            job.status = "failed"
            job.progress_stage = "failed"
            job.error = failure_from_output(output, run_id=job.run_id)
            # 診断用に結果も保持（/result は failed では 500）
            job.result = payload
        else:
            job.status = "succeeded"
            job.progress_stage = "done"
            job.result = payload

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
                owner_key_id=req.get("owner_key_id"),
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
                    job.finished_at = _utc_now()
                    self._finalize_success_or_output_error(job, result)
                self._hash_index.pop(job.input_hash, None)
                self._prune_jobs_unlocked()
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
                self._prune_jobs_unlocked()
        finally:
            try:
                from app.services.artifact_policy import maybe_cleanup_artifacts

                maybe_cleanup_artifacts()
            except Exception:
                logger.exception("artifact cleanup after job failed: %s", run_id)


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
    "failure_from_output",
    "get_run_manager",
    "output_indicates_failure",
    "set_run_manager",
]

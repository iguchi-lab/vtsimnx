"""
シミュレーション実行コア（同期 /run と非同期 /runs で共有）。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.builder import build_config_with_warning_details
from app.builder.logger import use_builder_log_file
from app.builder.validate import ConfigFileError, ValidationError
from app import solver_runner as solver_runner_mod
from app.solver_runner import (
    attach_builder_log_to_artifacts,
    attach_log_tail_to_output,
    cleanup_run_workdir,
    force_log_verbosity,
    run_workdir,
    write_artifact_manifest,
)
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]

# テストが main.run_solver を差し替えたときに使う（solver_runner 本体は汚さない）
_run_solver_hook: Optional[Callable[..., Dict[str, Any]]] = None


def _call_run_solver(built_config: Dict[str, Any], *, run_id: str, cancel_event: Optional[Any]) -> Dict[str, Any]:
    fn = _run_solver_hook or solver_runner_mod.run_solver
    try:
        return fn(
            built_config,
            run_id=run_id,
            write_manifest=False,
            cancel_event=cancel_event,
        )
    except TypeError:
        return fn(built_config)


@dataclass
class SimulationResult:
    run_id: str
    output: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    warning_details: List[Dict[str, Any]] = field(default_factory=list)


def execute_simulation(
    raw_config: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    debug: bool = False,
    debug_verbosity: int = 2,
    add_surface: Optional[bool] = None,
    add_aircon: Optional[bool] = None,
    add_capacity: Optional[bool] = None,
    add_moisture_capacity: Optional[bool] = None,
    add_surface_solar: Optional[bool] = None,
    add_surface_nocturnal: Optional[bool] = None,
    add_surface_radiation: Optional[bool] = None,
    add_surface_radiation_exclude_glass: Optional[bool] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[Any] = None,
    keep_artifacts: bool = True,
) -> SimulationResult:
    """
    builder → solver → artifact 後処理を実行する。
    HTTP / ジョブ層から独立したコア。
    """
    rid = run_id or uuid.uuid4().hex
    run_dir = run_workdir(rid)
    builder_log_tmp = run_dir / "builder.log.tmp"
    api_t0 = time.perf_counter()

    def _progress(stage: str, message: str = "") -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, message)
            except Exception:
                pass

    try:
        _progress("builder", "building config")
        build_stats_out: list = []
        builder_t0 = time.perf_counter()
        with use_builder_log_file(builder_log_tmp):
            built_config, warnings, warning_details = build_config_with_warning_details(
                raw_config,
                output_path=None,
                add_surface=add_surface,
                add_aircon=add_aircon,
                add_capacity=add_capacity,
                add_moisture_capacity=add_moisture_capacity,
                add_surface_solar=add_surface_solar,
                add_surface_nocturnal=add_surface_nocturnal,
                add_surface_radiation=add_surface_radiation,
                add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
                build_stats_out=build_stats_out,
            )
        builder_t1 = time.perf_counter()

        force_log_verbosity(built_config, debug=debug, debug_verbosity=debug_verbosity, default_verbosity=1)

        _progress("solver", "running solver")
        solver_t0 = time.perf_counter()
        output = _call_run_solver(built_config, run_id=rid, cancel_event=cancel_event)
        solver_t1 = time.perf_counter()

        _progress("postprocess", "attaching artifacts")
        artifact_t0 = time.perf_counter()
        attach_builder_log_to_artifacts(
            output,
            builder_log_path=builder_log_tmp,
            artifact_filename="builder.log",
            delete_source=True,
            build_config=built_config,
        )
        status = output.get("status")
        has_error = isinstance(output.get("error"), str) and bool(str(output.get("error")).strip())
        if has_error or (isinstance(status, str) and status.lower() == "error"):
            attach_log_tail_to_output(output)
        write_artifact_manifest(output)
        artifact_t1 = time.perf_counter()

        api_t1 = time.perf_counter()
        output["api_timings"] = {
            "builder_ms": (builder_t1 - builder_t0) * 1000.0,
            "solver_ms": (solver_t1 - solver_t0) * 1000.0,
            "artifact_postprocess_ms": (artifact_t1 - artifact_t0) * 1000.0,
            "api_total_ms": (api_t1 - api_t0) * 1000.0,
        }
        return SimulationResult(
            run_id=rid,
            output=output,
            warnings=warnings,
            warning_details=warning_details,
        )
    finally:
        cleanup_run_workdir(rid, keep_artifacts=keep_artifacts)


__all__ = ["SimulationResult", "execute_simulation", "ValidationError", "ConfigFileError"]

"""
シミュレーション実行サービス（builder → solver → artifact 後処理）。

テストは本モジュールの `run_solver` / `build_config_with_warning_details` を差し替え可能。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.builder import build_config_with_warning_details as _default_build
from app.builder.logger import use_builder_log_file
from app.builder.validate import ConfigFileError, ValidationError
from app.solver_runner import (
    attach_builder_log_to_artifacts,
    attach_log_tail_to_output,
    cleanup_run_workdir,
    force_log_verbosity,
    resolve_artifact_path,
    run_workdir,
    write_artifact_manifest,
)
from app.solver_runner import run_solver as _default_run_solver
from app.schemas.config import UnknownKeysMode, prepare_raw_config
from app.schemas.request import SimulationRequest
from app.schemas.response import SimulationResponse
from app.services.artifact_policy import (
    enforce_run_size_limit,
    mark_run_active,
    mark_run_inactive,
    write_owner_metadata,
)
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]

# テスト差し替え用（モジュール属性）
run_solver = _default_run_solver
build_config_with_warning_details = _default_build


@dataclass
class SimulationResult:
    run_id: str
    output: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    warning_details: List[Dict[str, Any]] = field(default_factory=list)


def _call_run_solver(built_config: Dict[str, Any], *, run_id: str, cancel_event: Optional[Any]) -> Dict[str, Any]:
    fn = run_solver
    try:
        return fn(
            built_config,
            run_id=run_id,
            write_manifest=False,
            cancel_event=cancel_event,
        )
    except TypeError:
        return fn(built_config)


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
    unknown_keys: str = "strip",
    initial_warnings: Optional[List[str]] = None,
    initial_warning_details: Optional[List[Dict[str, Any]]] = None,
    owner_key_id: Optional[str] = None,
) -> SimulationResult:
    """builder → solver → artifact 後処理を実行する。"""
    rid = run_id or uuid.uuid4().hex
    run_dir = run_workdir(rid)
    builder_log_tmp = run_dir / "builder.log.tmp"
    api_t0 = time.perf_counter()
    mark_run_active(rid)

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
                unknown_keys=unknown_keys,
            )
        builder_t1 = time.perf_counter()

        if initial_warnings:
            warnings = list(initial_warnings) + list(warnings)
        if initial_warning_details:
            warning_details = list(initial_warning_details) + list(warning_details)

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

        artifact_name = output.get("artifact_dir")
        if isinstance(artifact_name, str) and artifact_name:
            art_path = resolve_artifact_path(artifact_name)
            if art_path is not None:
                write_owner_metadata(art_path, key_id=owner_key_id, run_id=rid)
                try:
                    enforce_run_size_limit(art_path)
                except RuntimeError as e:
                    output["status"] = "error"
                    output["error"] = str(e)

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
        mark_run_inactive(rid)
        cleanup_run_workdir(rid, keep_artifacts=keep_artifacts)


def prepare_request_config(req: SimulationRequest) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """型付き config を builder 向け dict にし、未知キー警告を返す。"""
    return prepare_raw_config(req.config, unknown_keys=req.unknown_keys)


def run_from_request(req: SimulationRequest, *, owner_key_id: Optional[str] = None) -> SimulationResponse:
    """HTTP /run 相当の同期実行。"""
    raw_config, pre_warnings, pre_details = prepare_request_config(req)
    result = execute_simulation(
        raw_config,
        debug=req.debug,
        debug_verbosity=req.debug_verbosity,
        add_surface=req.add_surface,
        add_aircon=req.add_aircon,
        add_capacity=req.add_capacity,
        add_moisture_capacity=req.add_moisture_capacity,
        add_surface_solar=req.add_surface_solar,
        add_surface_nocturnal=req.add_surface_nocturnal,
        add_surface_radiation=req.add_surface_radiation,
        add_surface_radiation_exclude_glass=req.add_surface_radiation_exclude_glass,
        unknown_keys=req.unknown_keys,
        initial_warnings=pre_warnings,
        initial_warning_details=pre_details,
        owner_key_id=owner_key_id,
    )
    return SimulationResponse(
        result=result.output,
        warnings=result.warnings,
        warning_details=result.warning_details,
    )


def run_simulation_core(
    *,
    raw_config: Dict[str, Any],
    debug: bool,
    debug_verbosity: int,
    add_surface: Optional[bool] = None,
    add_aircon: Optional[bool] = None,
    add_capacity: Optional[bool] = None,
    add_moisture_capacity: Optional[bool] = None,
    add_surface_solar: Optional[bool] = None,
    add_surface_nocturnal: Optional[bool] = None,
    add_surface_radiation: Optional[bool] = None,
    add_surface_radiation_exclude_glass: Optional[bool] = None,
    unknown_keys: UnknownKeysMode = "strip",
) -> SimulationResponse:
    """CLI / テスト用の単発実行。"""
    prepared, pre_warnings, pre_details = prepare_raw_config(raw_config, unknown_keys=unknown_keys)
    result = execute_simulation(
        prepared,
        debug=debug,
        debug_verbosity=debug_verbosity,
        add_surface=add_surface,
        add_aircon=add_aircon,
        add_capacity=add_capacity,
        add_moisture_capacity=add_moisture_capacity,
        add_surface_solar=add_surface_solar,
        add_surface_nocturnal=add_surface_nocturnal,
        add_surface_radiation=add_surface_radiation,
        add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
        unknown_keys=unknown_keys,
        initial_warnings=pre_warnings,
        initial_warning_details=pre_details,
    )
    return SimulationResponse(
        result=result.output,
        warnings=result.warnings,
        warning_details=result.warning_details,
    )


def submit_run_payload(req: SimulationRequest) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """
    非同期ジョブ投入用に config を正規化し、request dict 用の警告を返す。
    """
    return prepare_request_config(req)


__all__ = [
    "ConfigFileError",
    "SimulationResult",
    "ValidationError",
    "build_config_with_warning_details",
    "execute_simulation",
    "prepare_request_config",
    "run_from_request",
    "run_simulation_core",
    "run_solver",
    "submit_run_payload",
]

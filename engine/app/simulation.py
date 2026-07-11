"""
後方互換: 旧 `app.simulation` からの import を維持する。
実装の正本は `app.services.simulation`。
"""
from app.services.simulation import (  # noqa: F401
    ConfigFileError,
    SimulationResult,
    ValidationError,
    build_config_with_warning_details,
    execute_simulation,
    prepare_request_config,
    run_from_request,
    run_simulation_core,
    run_solver,
    submit_run_payload,
)

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

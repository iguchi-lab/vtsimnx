"""
vtsimnx — 建築環境シミュレーション用 Python クライアント。

公開 API の安定性区分は ``vtsimnx.api_stability`` および ``docs/public_api.md`` を参照。
バージョン正本は ``pyproject.toml`` の ``project.version``（``vtsimnx.get_version()``）。
"""
from __future__ import annotations

import warnings
from typing import Any

from ._version import __version__, get_version
from . import units

# --- stable ---
from .run_calc import run_calc, CalcRunResult, RunCalcAPIError
from .artifacts import get_artifact_file, get_artifact_bytes

# --- experimental (top-level re-export; 変更しうる) ---
from .archenv import (
    sun_loc,
    make_wind,
    nocturnal_gain_by_angles,
    solar_gain_by_angles,
    solar_gain_by_angles_with_shade,
    ground_temperature_by_depth,
    calc_PMV,
    calc_PPD,
    calc_fungal_index,
)
from .utils.utils import read_json, read_csv, index, read_hasp
from .materials import materials as materials
from . import schedule as schedule

# deprecated top-level names are resolved lazily in __getattr__

__all_stable__ = [
    "__version__",
    "get_version",
    "units",
    "run_calc",
    "CalcRunResult",
    "RunCalcAPIError",
    "get_artifact_file",
    "get_artifact_bytes",
]

__all_experimental__ = [
    "sun_loc",
    "make_wind",
    "nocturnal_gain_by_angles",
    "solar_gain_by_angles",
    "solar_gain_by_angles_with_shade",
    "ground_temperature_by_depth",
    "calc_PMV",
    "calc_PPD",
    "calc_fungal_index",
    "read_json",
    "read_csv",
    "index",
    "read_hasp",
    "materials",
    "schedule",
]

__all_deprecated__ = [
    "make_8760_data",
    "ac_mode",
    "pre_tmp",
    "pre_rh",
    "vol",
    "sensible_heat",
]

__all__ = [
    *__all_stable__,
    *__all_experimental__,
    *__all_deprecated__,
]

_DEPRECATED_IMPORTS: dict[str, tuple[str, str, str]] = {
    # name: (module, attr, remove_in)
    "make_8760_data": ("vtsimnx.schedule", "make_8760_data", "2.0.0"),
    "ac_mode": ("vtsimnx.schedule", "ac_mode", "2.0.0"),
    "pre_tmp": ("vtsimnx.schedule", "pre_tmp", "2.0.0"),
    "pre_rh": ("vtsimnx.schedule", "pre_rh", "2.0.0"),
    "vol": ("vtsimnx.schedule", "vol", "2.0.0"),
    "sensible_heat": ("vtsimnx.schedule", "sensible_heat", "2.0.0"),
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_IMPORTS:
        mod_name, attr, remove_in = _DEPRECATED_IMPORTS[name]
        warnings.warn(
            f"vtsimnx.{name} is deprecated and will be removed in {remove_in}; "
            f"use {mod_name}.{attr} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'vtsimnx' has no attribute {name!r}")

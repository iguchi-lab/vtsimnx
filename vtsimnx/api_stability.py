"""公開 API の安定性区分。

- stable: セマンティックバージョンで互換を保証
- experimental: 予告なく変更しうる
- deprecated: 廃止予定（``deprecated_in`` / ``remove_in``）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ApiSymbol:
    name: str
    tier: str  # stable | experimental | deprecated
    import_path: str
    returns: str = ""
    units_note: str = ""
    deprecated_in: str | None = None
    remove_in: str | None = None
    replacement: str | None = None


STABLE: Final[tuple[ApiSymbol, ...]] = (
    ApiSymbol("run_calc", "stable", "vtsimnx.run_calc", returns="CalcRunResult"),
    ApiSymbol("CalcRunResult", "stable", "vtsimnx.run_calc"),
    ApiSymbol("RunCalcAPIError", "stable", "vtsimnx.run_calc"),
    ApiSymbol("get_artifact_file", "stable", "vtsimnx.artifacts", returns="Path | bytes"),
    ApiSymbol("__version__", "stable", "vtsimnx", returns="str"),
    ApiSymbol("get_version", "stable", "vtsimnx", returns="str"),
    ApiSymbol("units", "stable", "vtsimnx.units", returns="module"),
)

EXPERIMENTAL: Final[tuple[ApiSymbol, ...]] = (
    ApiSymbol("sun_loc", "experimental", "vtsimnx.archenv", returns="DataFrame/ndarray", units_note="deg"),
    ApiSymbol("make_wind", "experimental", "vtsimnx.archenv"),
    ApiSymbol("nocturnal_gain_by_angles", "experimental", "vtsimnx.archenv", units_note="W/m2"),
    ApiSymbol("solar_gain_by_angles", "experimental", "vtsimnx.archenv", units_note="W/m2"),
    ApiSymbol("solar_gain_by_angles_with_shade", "experimental", "vtsimnx.archenv", units_note="W/m2"),
    ApiSymbol("ground_temperature_by_depth", "experimental", "vtsimnx.archenv", units_note="degC"),
    ApiSymbol("calc_PMV", "experimental", "vtsimnx.archenv", returns="float"),
    ApiSymbol("calc_PPD", "experimental", "vtsimnx.archenv", returns="float"),
    ApiSymbol("calc_fungal_index", "experimental", "vtsimnx.archenv", returns="float", units_note="RH %"),
    ApiSymbol("columns", "experimental", "vtsimnx.archenv.columns", returns="module"),
    ApiSymbol("materials", "experimental", "vtsimnx.materials"),
    ApiSymbol("schedule", "experimental", "vtsimnx.schedule", returns="module"),
    ApiSymbol("read_json", "experimental", "vtsimnx.utils"),
    ApiSymbol("read_csv", "experimental", "vtsimnx.utils"),
    ApiSymbol("index", "experimental", "vtsimnx.utils"),
    ApiSymbol("read_hasp", "experimental", "vtsimnx.utils"),
)

DEPRECATED: Final[tuple[ApiSymbol, ...]] = (
    ApiSymbol(
        "make_8760_data",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.make_8760_data",
    ),
    ApiSymbol(
        "ac_mode",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.ac_mode",
    ),
    ApiSymbol(
        "pre_tmp",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.pre_tmp",
    ),
    ApiSymbol(
        "pre_rh",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.pre_rh",
    ),
    ApiSymbol(
        "vol",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.vol",
        units_note="m3/s or schedule-specific",
    ),
    ApiSymbol(
        "sensible_heat",
        "deprecated",
        "vtsimnx.schedule",
        deprecated_in="1.2.0",
        remove_in="2.0.0",
        replacement="vtsimnx.schedule.sensible_heat",
        units_note="W",
    ),
)

__all__ = ["ApiSymbol", "STABLE", "EXPERIMENTAL", "DEPRECATED"]

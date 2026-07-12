"""入力スキーマ用単位定数（正本は vtsimnx.units / docs/units.md）。"""
from __future__ import annotations

from typing import Any, Callable

try:
    from vtsimnx import units as _units
except Exception:  # pragma: no cover

    class _FallbackUnits:
        TEMPERATURE_C = "degC"
        PRESSURE_PA = "Pa"
        VOLUME_M3 = "m3"
        VOLUME_FLOW_M3_S = "m3/s"
        MASS_FLOW_KG_S = "kg/s"
        HEAT_RATE_W = "W"
        CONDUCTANCE_W_K = "W/K"
        U_VALUE_W_M2K = "W/(m2·K)"
        AREA_M2 = "m2"
        LENGTH_M = "m"
        HUMIDITY_RATIO = "kg/kg'"
        MOISTURE_GEN_KG_S = "kg/s"
        TIME_S = "s"
        CONCENTRATION = "-"
        THERMAL_MASS_J_K = "J/K"
        SOLAR_IRRADIANCE_W_M2 = "W/m2"

        @staticmethod
        def field_extra(unit: str, **more: object) -> dict[str, Any]:
            out: dict[str, Any] = {"unit": unit}
            out.update(more)
            return out

    _units = _FallbackUnits()  # type: ignore[assignment]

TEMPERATURE_C = _units.TEMPERATURE_C
PRESSURE_PA = _units.PRESSURE_PA
VOLUME_M3 = _units.VOLUME_M3
VOLUME_FLOW_M3_S = _units.VOLUME_FLOW_M3_S
MASS_FLOW_KG_S = _units.MASS_FLOW_KG_S
HEAT_RATE_W = _units.HEAT_RATE_W
CONDUCTANCE_W_K = _units.CONDUCTANCE_W_K
U_VALUE_W_M2K = _units.U_VALUE_W_M2K
AREA_M2 = _units.AREA_M2
LENGTH_M = _units.LENGTH_M
HUMIDITY_RATIO = _units.HUMIDITY_RATIO
MOISTURE_GEN_KG_S = _units.MOISTURE_GEN_KG_S
TIME_S = _units.TIME_S
CONCENTRATION = _units.CONCENTRATION
THERMAL_MASS_J_K = _units.THERMAL_MASS_J_K
SOLAR_IRRADIANCE_W_M2 = _units.SOLAR_IRRADIANCE_W_M2
field_extra: Callable[..., dict[str, Any]] = _units.field_extra

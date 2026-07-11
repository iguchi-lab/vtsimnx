"""入力スキーマ用単位定数（正本は vtsimnx.units / docs/units.md）。"""
from __future__ import annotations

try:
    from vtsimnx.units import (  # type: ignore
        AREA_M2,
        CONCENTRATION,
        CONDUCTANCE_W_K,
        HEAT_RATE_W,
        HUMIDITY_RATIO,
        LENGTH_M,
        MASS_FLOW_KG_S,
        MOISTURE_GEN_KG_S,
        PRESSURE_PA,
        SOLAR_IRRADIANCE_W_M2,
        TEMPERATURE_C,
        THERMAL_MASS_J_K,
        TIME_S,
        U_VALUE_W_M2K,
        VOLUME_FLOW_M3_S,
        VOLUME_M3,
        field_extra,
    )
except Exception:  # pragma: no cover
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

    def field_extra(unit: str, **more: object) -> dict[str, object]:
        out: dict[str, object] = {"unit": unit}
        out.update(more)
        return out

"""単位系の明示（建築シミュレーション向け）。

入力 JSON / 成果物系列で使う SI 系単位をコード上でも参照できるようにする。
スキーマの Field(json_schema_extra={"unit": ...}) と docs/units.md の正本。
"""
from __future__ import annotations

from typing import Final, Mapping

# --- 基本単位（正本） ---
TEMPERATURE_C: Final = "degC"
PRESSURE_PA: Final = "Pa"
VOLUME_M3: Final = "m3"
VOLUME_FLOW_M3_S: Final = "m3/s"
VOLUME_FLOW_M3_H: Final = "m3/h"  # 一部入力（ドキュメント上の表記）で使用
MASS_FLOW_KG_S: Final = "kg/s"
HEAT_RATE_W: Final = "W"
CONDUCTANCE_W_K: Final = "W/K"
U_VALUE_W_M2K: Final = "W/(m2·K)"
AREA_M2: Final = "m2"
LENGTH_M: Final = "m"
HUMIDITY_RATIO: Final = "kg/kg'"  # 絶対湿度（乾き空気基準）
MOISTURE_GEN_KG_S: Final = "kg/s"
TIME_S: Final = "s"
CONCENTRATION: Final = "-"  # モデル依存（無次元または kg/m3 等）
THERMAL_MASS_J_K: Final = "J/K"
SOLAR_IRRADIANCE_W_M2: Final = "W/m2"

# フィールド名 → 単位
FIELD_UNITS: Mapping[str, str] = {
    "t": TEMPERATURE_C,
    "pre_temp": TEMPERATURE_C,
    "p": PRESSURE_PA,
    "v": VOLUME_M3,
    "vol": VOLUME_FLOW_M3_S,  # solver/builder 内部は m3/s を基本とする
    "x": HUMIDITY_RATIO,
    "c": CONCENTRATION,
    "conductance": CONDUCTANCE_W_K,
    "u_value": U_VALUE_W_M2K,
    "area": AREA_M2,
    "alpha": "-",
    "h_from": LENGTH_M,
    "h_to": LENGTH_M,
    "thermal_mass": THERMAL_MASS_J_K,
    "heat_generation": HEAT_RATE_W,
    "humidity_generation": MOISTURE_GEN_KG_S,
    "generation_rate": MOISTURE_GEN_KG_S,
    "solar": SOLAR_IRRADIANCE_W_M2,
    "nocturnal": SOLAR_IRRADIANCE_W_M2,
    "timestep": TIME_S,
}

# 成果物 series 名 → 単位
SERIES_UNITS: Mapping[str, str] = {
    "vent_pressure": PRESSURE_PA,
    "vent_flow_rate": VOLUME_FLOW_M3_S,
    "thermal_temperature": TEMPERATURE_C,
    "thermal_heat_rate_advection": HEAT_RATE_W,
    "thermal_heat_rate_heat_generation": HEAT_RATE_W,
    "thermal_heat_rate_solar_gain": HEAT_RATE_W,
    "thermal_heat_rate_nocturnal_loss": HEAT_RATE_W,
    "thermal_heat_rate_convection": HEAT_RATE_W,
    "thermal_heat_rate_conduction": HEAT_RATE_W,
    "thermal_heat_rate_radiation": HEAT_RATE_W,
    "thermal_heat_rate_capacity": HEAT_RATE_W,
    "humidity_x": HUMIDITY_RATIO,
    "humidity_flux": MASS_FLOW_KG_S,
    "concentration_c": CONCENTRATION,
    "concentration_flux": MASS_FLOW_KG_S,
    "aircon_sensible_heat": HEAT_RATE_W,
    "aircon_latent_heat": HEAT_RATE_W,
    "aircon_power": HEAT_RATE_W,
    "aircon_cop": "-",
}


def unit_for_field(name: str) -> str | None:
    return FIELD_UNITS.get(name)


def unit_for_series(name: str) -> str | None:
    return SERIES_UNITS.get(name)


def field_extra(unit: str, **more: object) -> dict[str, object]:
    """Pydantic Field(json_schema_extra=...) 用。"""
    out: dict[str, object] = {"unit": unit}
    out.update(more)
    return out


__all__ = [
    "TEMPERATURE_C",
    "PRESSURE_PA",
    "VOLUME_M3",
    "VOLUME_FLOW_M3_S",
    "VOLUME_FLOW_M3_H",
    "MASS_FLOW_KG_S",
    "HEAT_RATE_W",
    "CONDUCTANCE_W_K",
    "U_VALUE_W_M2K",
    "AREA_M2",
    "LENGTH_M",
    "HUMIDITY_RATIO",
    "MOISTURE_GEN_KG_S",
    "TIME_S",
    "CONCENTRATION",
    "THERMAL_MASS_J_K",
    "SOLAR_IRRADIANCE_W_M2",
    "FIELD_UNITS",
    "SERIES_UNITS",
    "unit_for_field",
    "unit_for_series",
    "field_extra",
]

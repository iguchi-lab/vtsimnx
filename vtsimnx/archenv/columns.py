"""archenv DataFrame/Series 列名の正本（英語キー）。

日本語列名は後方互換用エイリアスとして ``TO_JP`` / ``with_japanese_column_aliases`` で付与できる。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

# --- solar position ---
DAY_OF_YEAR_N = "day_of_year_n"
HOUR_H = "hour_h"
SOLAR_DECLINATION_DEG = "solar_declination_deg"
EQUATION_OF_TIME_H = "equation_of_time_h"
HOUR_ANGLE_DEG = "hour_angle_deg"
SIN_SOLAR_ALTITUDE = "sin_solar_altitude"
COS_SOLAR_ALTITUDE = "cos_solar_altitude"
SOLAR_ALTITUDE_DEG = "solar_altitude_deg"
SIN_SOLAR_AZIMUTH = "sin_solar_azimuth"
COS_SOLAR_AZIMUTH = "cos_solar_azimuth"
SOLAR_AZIMUTH_DEG = "solar_azimuth_deg"

# --- irradiance / separation ---
GHI = "ghi"
DHI = "dhi"
DNI = "dni"
CLEARNESS_INDEX_KT = "clearness_index_kt"

# --- surface solar gain ---
COS_INCIDENCE = "cos_incidence"
BEAM_ON_SURFACE = "beam_on_surface"
BEAM_ON_GLASS = "beam_on_glass"
DIFFUSE_SKY_ON_SURFACE = "diffuse_sky_on_surface"
DIFFUSE_GROUND_REFLECTED = "diffuse_ground_reflected"
DIFFUSE_SKY_ON_GLASS = "diffuse_sky_on_glass"
DIFFUSE_GROUND_REFLECTED_GLASS = "diffuse_ground_reflected_glass"
SOLAR_GAIN_WALL = "solar_gain_wall"
SOLAR_GAIN_GLASS = "solar_gain_glass"
SOLAR_GAIN = "solar_gain"
SHADE_RATIO = "shade_ratio"
SUNLIT_RATIO = "sunlit_ratio"

# --- nocturnal / ground / wind ---
NOCTURNAL_RADIATION = "nocturnal_radiation"
NOCTURNAL_RADIATION_HORIZONTAL = "nocturnal_radiation_horizontal"
GROUND_TEMPERATURE = "ground_temperature"
SURFACE_EQUIVALENT_TEMPERATURE = "surface_equivalent_temperature"

WIND_SPEED_E = "wind_speed_e"
WIND_SPEED_S = "wind_speed_s"
WIND_SPEED_W = "wind_speed_w"
WIND_SPEED_N = "wind_speed_n"
WIND_PRESSURE_E = "wind_pressure_e"
WIND_PRESSURE_S = "wind_pressure_s"
WIND_PRESSURE_W = "wind_pressure_w"
WIND_PRESSURE_N = "wind_pressure_n"
WIND_PRESSURE_H = "wind_pressure_h"


def ground_temperature_at_depth(depth_m: float) -> str:
    return f"ground_temperature_{float(depth_m):.3f}m"


# English -> legacy Japanese (documentation / migration)
TO_JP: Mapping[str, str] = {
    DAY_OF_YEAR_N: "元日からの通し日数 N",
    HOUR_H: "時刻 H",
    SOLAR_DECLINATION_DEG: "太陽の赤緯 delta_d",
    EQUATION_OF_TIME_H: "太陽の均時差 e_d",
    HOUR_ANGLE_DEG: "太陽の時角 T_d_t",
    SIN_SOLAR_ALTITUDE: "太陽高度の正弦 sin_hs",
    COS_SOLAR_ALTITUDE: "太陽高度の余弦 cos_hs",
    SOLAR_ALTITUDE_DEG: "太陽高度 hs",
    SIN_SOLAR_AZIMUTH: "太陽方位角の正弦 sin_AZs",
    COS_SOLAR_AZIMUTH: "太陽方位角の余弦 cos_AZs",
    SOLAR_AZIMUTH_DEG: "太陽方位角 AZs",
    GHI: "水平面全天日射量",
    DHI: "水平面拡散日射量 Id",
    DNI: "法線面直達日射量 Ib",
    CLEARNESS_INDEX_KT: "晴天指数 Kt",
    COS_INCIDENCE: "入射角cos",
    BEAM_ON_SURFACE: "直達日射量の面成分 Ib",
    BEAM_ON_GLASS: "直達日射量の面成分（ガラス） Ib_g",
    DIFFUSE_SKY_ON_SURFACE: "水平面拡散日射量の拡散成分",
    DIFFUSE_GROUND_REFLECTED: "水平面拡散日射量の反射成分",
    DIFFUSE_SKY_ON_GLASS: "水平面拡散日射量の拡散成分（ガラス）",
    DIFFUSE_GROUND_REFLECTED_GLASS: "水平面拡散日射量の反射成分（ガラス）",
    SOLAR_GAIN_WALL: "日射熱取得量（壁面）",
    SOLAR_GAIN_GLASS: "日射熱取得量（ガラス）",
    SOLAR_GAIN: "日射熱取得量",
    SHADE_RATIO: "被影率η",
    SUNLIT_RATIO: "日向率(1-η)",
    NOCTURNAL_RADIATION: "夜間放射量",
    NOCTURNAL_RADIATION_HORIZONTAL: "夜間放射量_水平",
    GROUND_TEMPERATURE: "地盤温度",
    SURFACE_EQUIVALENT_TEMPERATURE: "地表等価温度",
    WIND_SPEED_E: "風速_E",
    WIND_SPEED_S: "風速_S",
    WIND_SPEED_W: "風速_W",
    WIND_SPEED_N: "風速_N",
    WIND_PRESSURE_E: "風圧_E",
    WIND_PRESSURE_S: "風圧_S",
    WIND_PRESSURE_W: "風圧_W",
    WIND_PRESSURE_N: "風圧_N",
    WIND_PRESSURE_H: "風圧_H",
}

# sep_direct_diffuse で一時的に使っていた別名
TO_JP_EXTRA: Mapping[str, str] = {
    SOLAR_ALTITUDE_DEG: "太陽高度",  # 旧 sep 出力の別名（正本は 太陽高度 hs）
}


def with_japanese_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """英語列に加え、既知の日本語エイリアス列を複製して返す（移行用）。"""
    out = df.copy()
    for en, jp in TO_JP.items():
        if en in out.columns and jp not in out.columns:
            out[jp] = out[en]
    return out


def rename_to_japanese(df: pd.DataFrame) -> pd.DataFrame:
    """英語列名を日本語レガシー名へリネーム（存在する列のみ）。"""
    mapping = {en: jp for en, jp in TO_JP.items() if en in df.columns}
    return df.rename(columns=mapping)

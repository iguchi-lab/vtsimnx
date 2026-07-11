"""風向・風速から方位別風圧を算出する。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import columns as C
from .archenv import air_density

_WIND_DIRS = (
    ("E", C.WIND_SPEED_E, C.WIND_PRESSURE_E),
    ("S", C.WIND_SPEED_S, C.WIND_PRESSURE_S),
    ("W", C.WIND_SPEED_W, C.WIND_PRESSURE_W),
    ("N", C.WIND_SPEED_N, C.WIND_PRESSURE_N),
)


def make_wind(
    d: pd.Series,
    s: pd.Series,
    c_in: float = 0.7,
    c_out: float = -0.55,
    c_horizontal: float = -0.90,
    *,
    air_density_kg_m3: float | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """風向・風速から各方位の風圧を算出する。

    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数
      air_density_kg_m3: 空気密度 [kg/m3]。None なら 20degC 近似。

    戻り値:
      (DataFrame, dict[str, Series]) 中間列と方位別風圧（E/S/W/N/H）
      列名は英語キー（``vtsimnx.archenv.columns``）。
    """
    if not isinstance(d, pd.Series) or not isinstance(s, pd.Series):
        raise TypeError("d と s は pandas.Series で指定してください。")
    if not d.index.equals(s.index):
        raise ValueError("d と s の index は一致している必要があります。")
    if d.isna().any() or s.isna().any():
        raise ValueError("d と s に NaN は指定できません。")
    if (s < 0).any():
        raise ValueError("風速 s は 0 以上である必要があります。")

    d_num = pd.to_numeric(d, errors="coerce")
    if d_num.isna().any():
        raise TypeError("風向カテゴリ d は数値（0..16）で指定してください。")
    if ((d_num < 0) | (d_num > 16)).any():
        raise ValueError("風向カテゴリ d は 0..16 の範囲で指定してください。")

    d = d_num.astype("float64")
    s = s.astype("float64")
    rho = float(air_density(20.0) if air_density_kg_m3 is None else air_density_kg_m3)
    half_rho = rho / 2.0

    ang = np.radians(d * 22.5)
    sin_a = np.sin(ang)
    cos_a = np.cos(ang)

    df = pd.DataFrame(index=d.index)
    # E: +sin, S: -cos, W: -sin, N: +cos
    components = {
        "E": sin_a * s,
        "S": -cos_a * s,
        "W": -sin_a * s,
        "N": cos_a * s,
    }
    wind_pressure: dict[str, pd.Series] = {}
    for key, speed_col, press_col in _WIND_DIRS:
        speed = components[key]
        df[speed_col] = speed
        # 風速成分の符号で風上/風下係数を切替（既存互換）
        press = pd.Series(index=d.index, dtype="float64")
        pos = speed >= 0
        press.loc[pos] = half_rho * c_in * speed.loc[pos] ** 2
        press.loc[~pos] = -half_rho * c_out * speed.loc[~pos] ** 2
        df[press_col] = press
        wind_pressure[key] = press

    df[C.WIND_PRESSURE_H] = half_rho * c_horizontal * (s**2)
    wind_pressure["H"] = df[C.WIND_PRESSURE_H]
    return df, wind_pressure


__all__ = ["make_wind"]

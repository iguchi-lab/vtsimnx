"""夜間放射（長波）の推算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import columns as C
from .archenv import SIGMA_NOCTURNAL, MJ_to_Wh, to_kelvin, vapor_pressure_from_rh_pa


def rn(t: pd.Series | np.ndarray | float, h: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """夜間放射量 [MJ/m2] を推算する。

    t: 気温 [degC]
    h: 相対湿度 [%]
    """
    return (
        94.21
        + 39.06 * np.sqrt(vapor_pressure_from_rh_pa(t, h) / 100)
        - 0.85 * SIGMA_NOCTURNAL * np.power(to_kelvin(t), 4)
    ) * 4.187 / 1000


def nocturnal_gain_by_angles(
    *,
    tilt_deg: float,
    t_out: pd.Series | None = None,
    rh_out: pd.Series | None = None,
    rn_horizontal: pd.Series | None = None,
    return_details: bool = False,
) -> pd.DataFrame | pd.Series:
    """
    傾斜角だけ指定して、その面の夜間放射量（長波放射）[Wh/m2] を返す。

    入力（どちらか）:
      - t_out [degC] + rh_out [%]: rn(t,h) から水平面の夜間放射量を推算
      - rn_horizontal: 水平面の夜間放射量 [Wh/m2] を直接与える

    tilt_deg: 0=水平上向き, 90=鉛直
    """
    if not (0.0 <= float(tilt_deg) <= 180.0):
        raise ValueError("nocturnal_gain_by_angles: tilt_deg must be in [0, 180].")

    if rn_horizontal is None:
        if t_out is None or rh_out is None:
            raise TypeError("nocturnal_gain_by_angles: t_out/rh_out または rn_horizontal を指定してください。")
        if not isinstance(t_out, pd.Series) or not isinstance(rh_out, pd.Series):
            raise TypeError("nocturnal_gain_by_angles: t_out と rh_out は pandas.Series で指定してください。")
        if not t_out.index.equals(rh_out.index):
            raise ValueError("nocturnal_gain_by_angles: rh_out index must match t_out index.")
        if not isinstance(t_out.index, pd.DatetimeIndex):
            raise TypeError("nocturnal_gain_by_angles: t_out index must be DatetimeIndex.")
        rn_horizontal = MJ_to_Wh(rn(t_out, rh_out))
    else:
        if not isinstance(rn_horizontal, pd.Series):
            raise TypeError("nocturnal_gain_by_angles: rn_horizontal は pandas.Series で指定してください。")
        if not isinstance(rn_horizontal.index, pd.DatetimeIndex):
            raise TypeError("nocturnal_gain_by_angles: rn_horizontal index must be DatetimeIndex.")

    beta = np.radians(float(tilt_deg))
    f_sky = (1.0 + float(np.cos(beta))) / 2.0

    s_nocturnal = (rn_horizontal * f_sky).rename(C.NOCTURNAL_RADIATION)
    if not return_details:
        return s_nocturnal

    out = pd.DataFrame(index=rn_horizontal.index)
    out[C.NOCTURNAL_RADIATION_HORIZONTAL] = rn_horizontal
    out[C.NOCTURNAL_RADIATION] = s_nocturnal
    return out


__all__ = ["rn", "nocturnal_gain_by_angles"]

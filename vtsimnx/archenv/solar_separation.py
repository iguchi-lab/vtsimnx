"""直散分離（Erbs）。列名は英語キー。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import columns as C
from .archenv import MJ_to_Wh, Solar_I, Wh_to_MJ


def Kt(IG: Any, alt: Any) -> Any:
    """晴天指数 Kt。IG [MJ/m2], alt [deg]。"""
    return IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))


def _as_array(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def Id(IG: Any, kt: Any) -> np.ndarray:
    """水平面拡散日射量の推定（Erbs）。IG/kt は配列。"""
    IG_arr = _as_array(IG)
    kt_arr = _as_array(kt)
    s_Id = np.zeros(len(kt_arr), dtype="float64")
    m1 = kt_arr <= 0.22
    m2 = (0.22 < kt_arr) & (kt_arr <= 0.80)
    m3 = 0.80 < kt_arr

    s_Id[m1] = IG_arr[m1] * (1 - 0.09 * kt_arr[m1])
    k2 = kt_arr[m2]
    s_Id[m2] = IG_arr[m2] * (
        0.9511 - 0.1604 * k2 + 4.388 * np.power(k2, 2) - 16.638 * np.power(k2, 3) + 12.336 * np.power(k2, 4)
    )
    s_Id[m3] = 0.365 * IG_arr[m3]
    return s_Id


def Ib(IG: Any, Id_val: Any, alt: Any, min_alt_deg: float = 0.0) -> np.ndarray:
    """法線面直達日射量の推定。"""
    IG_arr = _as_array(IG)
    Id_arr = _as_array(Id_val)
    alt_arr = _as_array(alt)
    s_Ib = np.zeros(len(Id_arr), dtype="float64")
    sin_alt = np.sin(np.radians(alt_arr))
    valid = (alt_arr > float(min_alt_deg)) & (sin_alt > 0.0)
    s_Ib[valid] = (IG_arr[valid] - Id_arr[valid]) / sin_alt[valid]
    cap = valid & (alt_arr < 10.0) & (s_Ib > IG_arr)
    s_Ib[cap] = IG_arr[cap]
    return s_Ib


def sep_direct_diffuse(
    s_ig: pd.Series,
    s_hs: pd.Series,
    min_sun_alt_deg: float = 0.0,
) -> pd.DataFrame:
    """全天日射量と太陽高度から直散分離（Erbs）を行い Kt/DHI/DNI を返す。"""
    df = pd.DataFrame(index=s_ig.index)
    df[C.GHI] = s_ig.astype("float64")
    df[C.SOLAR_ALTITUDE_DEG] = s_hs.astype("float64")
    df[C.CLEARNESS_INDEX_KT] = Kt(Wh_to_MJ(df[C.GHI]), df[C.SOLAR_ALTITUDE_DEG])
    df[C.DHI] = MJ_to_Wh(Id(Wh_to_MJ(df[C.GHI]), df[C.CLEARNESS_INDEX_KT]))
    df[C.DNI] = MJ_to_Wh(
        Ib(
            Wh_to_MJ(df[C.GHI]),
            Wh_to_MJ(df[C.DHI]),
            df[C.SOLAR_ALTITUDE_DEG],
            min_alt_deg=min_sun_alt_deg,
        )
    )
    return df


__all__ = ["Kt", "Id", "Ib", "sep_direct_diffuse"]

import numpy as np
import pandas as pd

from .archenv import Wh_to_MJ, MJ_to_Wh, Solar_I


# 直散分離（Erbs）
def Kt(IG, alt):
    """晴天指数 Kt。"""
    return IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))


def _as_array(x):
    """pandas Series を含む配列状入力を numpy 配列に正規化する"""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def Id(IG, kt):
    """水平面拡散日射量の推定（Erbs 法）
    IG: 水平面全天日射量 [Wh/m2] 配列
    kt: 晴天指数
    """
    IG_arr = _as_array(IG)
    kt_arr = _as_array(kt)
    s_Id = np.zeros(len(kt_arr), dtype="float64")
    m1 = kt_arr <= 0.22
    m2 = (0.22 < kt_arr) & (kt_arr <= 0.80)
    m3 = 0.80 < kt_arr

    s_Id[m1] = IG_arr[m1] * (1 - 0.09 * kt_arr[m1])
    k2 = kt_arr[m2]
    s_Id[m2] = IG_arr[m2] * (
        0.9511
        - 0.1604 * k2
        + 4.388 * np.power(k2, 2)
        - 16.638 * np.power(k2, 3)
        + 12.336 * np.power(k2, 4)
    )
    s_Id[m3] = 0.365 * IG_arr[m3]
    return s_Id


def Ib(IG, Id, alt, min_alt_deg: float = 0.0):
    """法線面直達日射量の推定
    IG: 水平面全天日射量 [Wh/m2]
    Id: 水平面拡散日射量 [Wh/m2]
    alt: 太陽高度 [deg]
    min_alt_deg: 太陽高度がこの値未満のとき直達を 0 扱い（スパイク抑制用）
    """
    IG_arr  = _as_array(IG)
    Id_arr  = _as_array(Id)
    alt_arr = _as_array(alt)
    s_Ib = np.zeros(len(Id_arr), dtype="float64")
    sin_alt = np.sin(np.radians(alt_arr))
    valid = (alt_arr > float(min_alt_deg)) & (sin_alt > 0.0)
    s_Ib[valid] = (IG_arr[valid] - Id_arr[valid]) / sin_alt[valid]

    # 低高度での異常値対策（既存互換）
    cap = valid & (alt_arr < 10.0) & (s_Ib > IG_arr)
    s_Ib[cap] = IG_arr[cap]
    return s_Ib


def sep_direct_diffuse(s_ig, s_hs, min_sun_alt_deg: float = 0.0):
    """全天日射量と太陽高度から直散分離（Erbs）を行い Kt/Id/Ib を返す"""
    df = pd.concat([s_ig, s_hs], axis = 1)
    df.columns = ['水平面全天日射量', '太陽高度']
    df['晴天指数 Kt'] = Kt(Wh_to_MJ(df['水平面全天日射量']), df['太陽高度'])
    df['水平面拡散日射量 Id'] = MJ_to_Wh(Id(Wh_to_MJ(df['水平面全天日射量']), df['晴天指数 Kt']))
    df['法線面直達日射量 Ib'] = MJ_to_Wh(
        Ib(
            Wh_to_MJ(df['水平面全天日射量']),
            Wh_to_MJ(df['水平面拡散日射量 Id']),
            df['太陽高度'],
            min_alt_deg=min_sun_alt_deg,
        )
    )
    return df


__all__ = ["Kt", "Id", "Ib", "sep_direct_diffuse"]


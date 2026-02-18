import numpy as np
import pandas as pd

from .archenv import Wh_to_MJ, MJ_to_Wh, Solar_I


# 直散分離（Erbs）
Kt = lambda IG, alt: IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))  # 晴天指数


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
    s_Id = np.zeros(len(kt_arr))
    for i, k in enumerate(kt_arr):
        if   k <= 0.22:                 s_Id[i] = IG_arr[i] * (1 - 0.09 * k)
        elif (0.22 < k) & (k <= 0.80):  s_Id[i] = IG_arr[i] * (0.9511 -  0.1604 * k \
                                                                  +  4.388  * np.power(k, 2) \
                                                                  - 16.638  * np.power(k, 3) \
                                                                  + 12.336  * np.power(k, 4))
        elif 0.80 < k:                  s_Id[i] = 0.365 * IG_arr[i]
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
    s_Ib = np.zeros(len(Id_arr))
    for i, idv in enumerate(Id_arr):
        alt_i = float(alt_arr[i])
        if alt_i <= float(min_alt_deg):
            s_Ib[i] = 0.0
            continue
        s = np.sin(np.radians(alt_i))
        # 数値的に極小な sin での発散を避ける
        if s <= 0.0:
            s_Ib[i] = 0.0
            continue

        s_Ib[i] = (IG_arr[i] - idv) / s
        # 低高度での異常値対策（既存互換）
        if (alt_i < 10.0) & (s_Ib[i] > IG_arr[i]):
            s_Ib[i] = IG_arr[i]
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


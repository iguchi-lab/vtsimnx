import pandas as pd
import numpy as np
import datetime

from .archenv import Wh_to_MJ, MJ_to_Wh, _az_deg_from_sin_cos, _alt_deg_from_sin
from .archenv import Solar_I


# 直散分離（Erbs）
Kt = lambda IG, alt: IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))  # 晴天指数


def Id(IG, kt):
    """水平面拡散日射量の推定（Erbs 法）
    IG: 水平面全天日射量 [Wh/m2] 配列
    kt: 晴天指数
    """
    s_Id = np.zeros(len(kt))
    for i, k in enumerate(kt):
        if   k <= 0.22:                 s_Id[i] = IG[i] * (1 - 0.09 * k)
        elif (0.22 < k) & (k <= 0.80):  s_Id[i] = IG[i] * (0.9511 -  0.1604 * k \
                                                                  +  4.388  * np.power(k, 2) \
                                                                  - 16.638  * np.power(k, 3) \
                                                                  + 12.336  * np.power(k, 4))
        elif 0.80 < k:                  s_Id[i] = 0.365 * IG[i]
    return s_Id


def Ib(IG, Id, alt):
    """法線面直達日射量の推定
    IG: 水平面全天日射量 [Wh/m2]
    Id: 水平面拡散日射量 [Wh/m2]
    alt: 太陽高度 [deg]
    """
    s_Ib = np.zeros(len(Id))
    for i, id in enumerate(Id):
        s_Ib[i] =  (IG[i] - Id[i]) / np.sin(np.radians(alt[i]))
        if (alt[i] < 10.0) & (s_Ib[i] > IG[i]):  s_Ib[i] = IG[i]
    return s_Ib


# 太陽位置の計算に用いる基本式
# 太陽の赤緯 δ [deg]（年周の余弦和による近似）
delta_d = lambda N: (180 / np.pi) * (0.006322 \
                                     - 0.405748 * np.cos(2 * np.pi * N / 366 + 0.153231) \
                                     - 0.005880 * np.cos(4 * np.pi * N / 366 + 0.207099) \
                                     - 0.003233 * np.cos(6 * np.pi * N / 366 + 0.620129))

# 太陽の均時差 ed [h]（平均太陽時と真太陽時の差）
e_d     = lambda N: -0.000279 + 0.122772 * np.cos(2 * np.pi * N / 366 + 1.498311) \
                              - 0.165458 * np.cos(4 * np.pi * N / 366 - 1.261546) \
                              - 0.005354 * np.cos(6 * np.pi * N / 366 - 1.1571)

# 太陽の時角 T_d_t [deg]（正午=0°、午前は負、午後は正）
T_d_t   = lambda H, ed, L       : (H  + ed - 12.0) * 15.0 + (L - 135.0)

# 角度[deg]の正弦・余弦
sin     = lambda v              : np.sin(np.radians(v))
cos     = lambda v              : np.cos(np.radians(v))

# 太陽高度 hs の正弦: sin(hs) = sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(時角)
sin_hs  = lambda L, dd, tdt     : sin(L) * sin(dd) + cos(L) * cos(dd) * cos(tdt)
# 太陽方位角 AZs の正弦/余弦（象限判定は arctan2 で別途実施）
sin_AZs = lambda dd, tdt, c_h   : cos(dd) * sin(tdt) / c_h
cos_AZs = lambda s_h, L, dd, c_h: (s_h * sin(L) - sin(dd)) / (c_h * cos(L))


def sun_loc(idx, lat = 36.00, lon = 140.00, td = -0.5):
    """太陽位置を簡易式で算出（赤緯/均時差/時角から）
    idx: DatetimeIndex
    lat, lon: 緯度・経度 [deg]
    td: ローカル時刻微調整 [h]
    """
    df = pd.DataFrame(index=idx)
    # 元日からの通し日数（正午寄せ）
    df['元日からの通し日数 N'] = [
        (i - datetime.datetime(i.year, 1, 1)).days + 1.5
        for i in idx
    ]
    # 小数時間（時＋分/60）に微調整 td を加える
    df['時刻 H'] = (
        idx.strftime("%H").astype('float64')
        + idx.strftime("%M").astype('float64') / 60
        + td
    )
    # 太陽の基本角
    df['太陽の赤緯 delta_d'] = delta_d(df['元日からの通し日数 N'])
    df['太陽の均時差 e_d']   = e_d(df['元日からの通し日数 N'])
    df['太陽の時角 T_d_t']   = T_d_t(df['時刻 H'], df['太陽の均時差 e_d'], lon)
    # 太陽高度
    df['太陽高度の正弦 sin_hs'] = sin_hs(
        lat, df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t']
    )
    df['太陽高度の余弦 cos_hs'] = np.sqrt(
        1 - np.power(df['太陽高度の正弦 sin_hs'], 2)
    )
    df['太陽高度 hs'] = _alt_deg_from_sin(df['太陽高度の正弦 sin_hs'])

    # 太陽方位角
    df['太陽方位角の正弦 sin_AZs'] = sin_AZs(
        df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t'], df['太陽高度の余弦 cos_hs']
    )
    df['太陽方位角の余弦 cos_AZs'] = cos_AZs(
        df['太陽高度の正弦 sin_hs'], lat, df['太陽の時角 T_d_t'], df['太陽高度の余弦 cos_hs']
    )

    df['太陽方位角 AZs'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs']
    )

    return df


def astro_sun_loc(idx, lat = '36 00 00.00', lon = '140 00 00.00', td = -0.5):
    """astropy を用いた太陽位置の高精度計算
    lat, lon は DMS 表記の文字列を想定（例: '36 00 00.00'）
    戻り値: 仰角/方位角の sin, cos と角度 [deg]
    """
    import astropy.time
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.coordinates import AltAz
    from astropy.coordinates import EarthLocation

    loc = EarthLocation(lat = lat, lon = lon)
    time = astropy.time.Time(idx) + (-9 + td ) * u.hour
    sun = get_sun(time).transform_to(AltAz(obstime = time, location = loc))

    df = pd.DataFrame(index = idx)

    # 太陽高度（仰角）
    df['太陽高度の正弦 sin_alt'] = np.array([np.sin(s.alt) for s in sun]).astype('float64')
    df['太陽高度の余弦 cos_alt'] = np.array([np.cos(s.alt) for s in sun]).astype('float64')
    df['太陽高度 alt'] = np.degrees(
        np.arcsin(df['太陽高度の正弦 sin_alt'])
    )

    # 太陽方位角
    df['太陽方位角の正弦 sin_az'] = np.array([np.sin(s.az) for s in sun]).astype('float64')
    df['太陽方位角の余弦 cos_az'] = np.array([np.cos(s.az) for s in sun]).astype('float64')

    df['太陽方位角 az'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_az'], df['太陽方位角の余弦 cos_az']
    )

    return df


def sep_direct_diffuse(s_ig, s_hs):
    """全天日射量と太陽高度から直散分離（Erbs）を行い Kt/Id/Ib を返す"""
    df = pd.concat([s_ig, s_hs], axis = 1)
    df.columns = ['水平面全天日射量', '太陽高度']
    df['晴天指数 Kt'] = Kt(Wh_to_MJ(df['水平面全天日射量']), df['太陽高度'])
    df['水平面拡散日射量 Id'] = MJ_to_Wh(Id(Wh_to_MJ(df['水平面全天日射量']), df['晴天指数 Kt']))
    df['法線面直達日射量 Ib'] = MJ_to_Wh(
        Ib(Wh_to_MJ(df['水平面全天日射量']), Wh_to_MJ(df['水平面拡散日射量 Id']), df['太陽高度'])
    )
    return df


def eta(c):
    """角度補正の近似多項式
    P(c) = 2.3920·c − 3.8636·c^3 + 3.7568·c^5 − 1.3968·c^7
    Horner 形で評価して数値安定性と可読性を両立。
    """
    c2 = c * c
    return c * (2.3920 + c2 * (-3.8636 + c2 * (3.7568 + c2 * (-1.3968))))


def direc_solar(s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs):
    """直達・拡散・反射から方位別日射量（壁/ガラス/水平）を算出する"""
    df = pd.concat(
        [s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs],
        axis = 1
    )
    df.columns = [
        '法線面直達日射量 Ib', '水平面拡散日射量 Id',
        '太陽高度の正弦 sin_hs', '太陽高度の余弦 cos_hs', '太陽高度 hs',
        '太陽方位角の正弦 sin_AZs', '太陽方位角の余弦 cos_AZs', '太陽方位角 AZs'
    ]

    cond = df['太陽高度 hs'] > 0
    az = df['太陽方位角 AZs']
    cos_hs = df['太陽高度の余弦 cos_hs']
    sin_hs = df['太陽高度の正弦 sin_hs']
    sin_az = df['太陽方位角の正弦 sin_AZs']
    cos_az = df['太陽方位角の余弦 cos_AZs']
    Ib_col = df['法線面直達日射量 Ib']

    df.loc[cond & (-180 < az) & (az < 0),   '直達日射量の東面成分 Ib_E'] = (
        -1 * Ib_col * cos_hs * sin_az
    )  # 東
    df.loc[cond & (-90  < az) & (az < 90),  '直達日射量の南面成分 Ib_S'] = (
           Ib_col * cos_hs * cos_az
    )  # 南
    df.loc[cond & (0    < az) & (az < 180), '直達日射量の西面成分 Ib_W'] = (
           Ib_col * cos_hs * sin_az
    )  # 西
    df.loc[cond & (-180 < az) & (az < -90), '直達日射量の北面成分 Ib_N'] = (
        -1 * Ib_col * cos_hs * cos_az
    )  # 北
    df.loc[cond & (  90 < az) & (az < 180), '直達日射量の北面成分 Ib_N'] = (
        -1 * Ib_col * cos_hs * cos_az
    )  # 北
    df.loc[cond, '直達日射量の水平面成分 Ib_H'] = (
        Ib_col * sin_hs
    )
    df.loc[cond, '水平面拡散日射量の反射成分 Id_R'] = (
        (df['水平面拡散日射量 Id'] + Ib_col) * sin_hs * 0.5 * 0.1
    )
    df['水平面拡散日射量の拡散成分 Id_D'] = df['水平面拡散日射量 Id'] * 0.5

    df = df.fillna(0)

    df['直達日射量の東面成分（ガラス） Ib_E_g'] = (
        df['直達日射量の東面成分 Ib_E'] * eta(-1 * cos_hs * sin_az)
    )  # 東
    df['直達日射量の南面成分（ガラス） Ib_S_g'] = (
        df['直達日射量の南面成分 Ib_S'] * eta(      cos_hs * cos_az)
    )  # 南
    df['直達日射量の西面成分（ガラス） Ib_W_g'] = (
        df['直達日射量の西面成分 Ib_W'] * eta(      cos_hs * sin_az)
    )  # 西
    df['直達日射量の北面成分（ガラス） Ib_N_g'] = (
        df['直達日射量の北面成分 Ib_N'] * eta(-1 * cos_hs * cos_az)
    )  # 北
    df['直達日射量の水平面成分（ガラス） Ib_H_g'] = (
        df['直達日射量の水平面成分 Ib_H'] * eta(sin_hs)
    )  # 水平
    df['水平面拡散日射量の反射成分（ガラス） Id_R_g'] = (
        df['水平面拡散日射量の反射成分 Id_R'] * 0.808
    )  # 透過率
    df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] = (
        df['水平面拡散日射量の拡散成分 Id_D'] * 0.808
    )  # 透過率

    df['日射熱取得量（東面）'] = (
        df['直達日射量の東面成分 Ib_E']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（南面）'] = (
        df['直達日射量の南面成分 Ib_S']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（西面）'] = (
        df['直達日射量の西面成分 Ib_W']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（北面）'] = (
        df['直達日射量の北面成分 Ib_N']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（水平面）'] = (
        df['直達日射量の水平面成分 Ib_H']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )

    df['日射熱取得量（東面ガラス）'] = (
        df['直達日射量の東面成分（ガラス） Ib_E_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（南面ガラス）'] = (
        df['直達日射量の南面成分（ガラス） Ib_S_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（西面ガラス）'] = (
        df['直達日射量の西面成分（ガラス） Ib_W_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（北面ガラス）'] = (
        df['直達日射量の北面成分（ガラス） Ib_N_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（水平面ガラス）'] = (
        df['直達日射量の水平面成分（ガラス） Ib_H_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )

    return df


def make_solar(**kwargs):
    """方位別日射量の総合算出
    指定方法:
      - 全天日射量を与えると内部で直散分離 → 方位別
      - 法線面直達日射量と水平面拡散日射量を与えるとそのまま方位別へ
    戻り値:
      (DataFrame) 中間列と方位別日射（壁/ガラス/水平）
    """
    lat = kwargs['緯度'] if '緯度' in kwargs else 36.00
    lon = kwargs['経度'] if '経度' in kwargs else 140.00

    if '全天日射量' in kwargs:
        s_ig = kwargs['全天日射量']
        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            delta = (s_ig.index[1] - s_ig.index[0])
            sec   = delta.seconds
            micro = delta.microseconds
            td    = sec + micro / 1000000 / 2 / 3600
        df = pd.concat(
            [s_ig, sun_loc(s_ig.index, lat = lat, lon = lon, td = td)],
            axis = 1
        )
        df = pd.concat(
            [df, sep_direct_diffuse(s_ig, df['太陽高度 hs'])],
            axis = 1
        )  # 直散分離結果
        df = direc_solar(
            df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],              # 方位別日射量
            df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
            df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs']
        )
    else:
        if '法線面直達日射量' in kwargs:
            s_ib = kwargs['法線面直達日射量']
        else:
            raise Exception('ERROR: 法線面直達日射量 s_ib がありません。')
        if '水平面拡散日射量' in kwargs:
            s_id = kwargs['水平面拡散日射量']
        else:
            raise Exception('ERROR: 水平面拡散日射量 s_id がありません。')
        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            delta = (s_ib.index[1] - s_ib.index[0])
            sec   = delta.seconds
            micro = delta.microseconds
            td    = - sec + micro / 1000000 / 2 / 3600
        df = pd.concat(
            [s_ib, s_id, sun_loc(s_ib.index, lat = lat, lon = lon, td = td)],
            axis = 1
        )
        df = direc_solar(
            df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],
            df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
            df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs']
        )

    return df

import pandas as pd
import numpy as np
import datetime

from .archenv import (
    Wh_to_MJ, MJ_to_Wh, Solar_I,
    _az_deg_from_sin_cos,
)

# 直散分離（Erbs）
Kt = lambda IG, alt: IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))  # 晴天指数

def Id(IG, kt):
    """水平面拡散日射量の推定（Erbs 法）"""
    s_Id = np.zeros(len(kt))
    for i, k in enumerate(kt):
        if   k <= 0.22:                 s_Id[i] = IG[i] * (1 - 0.09 * k)
        elif (0.22 < k) & (k <= 0.80):  s_Id[i] = IG[i] * (0.9511 -  0.1604 * k \
                                                                  +  4.388  * np.power(k, 2) \
                                                                  - 16.638  * np.power(k, 3) \
                                                                  + 12.336  * np.power(k, 4))
        elif 0.80 < k:                  s_Id[i] = 0.365 * IG[i]
    return s_Id

def Ib(IG, Id, alt):
    """法線面直達日射量の推定"""
    s_Ib = np.zeros(len(Id))
    for i, id in enumerate(Id):
        s_Ib[i] =  (IG[i] - Id[i]) / np.sin(np.radians(alt[i]))
        if (alt[i] < 10.0) & (s_Ib[i] > IG[i]):  s_Ib[i] = IG[i]
    return s_Ib

# 太陽位置の計算に用いる基本式
# 太陽の赤緯 δ [deg]（年周の余弦和による近似）
delta_d = lambda N: (180 / np.pi) * (0.006322 \
                                     - 0.405748 * np.cos(2 * np.pi * N / 366 + 0.153231) \
                                     - 0.005880 * np.cos(4 * np.pi * N / 366 + 0.207099) \
                                     - 0.003233 * np.cos(6 * np.pi * N / 366 + 0.620129))

# 太陽の均時差 ed [h]（平均太陽時と真太陽時の差）
e_d     = lambda N: -0.000279 + 0.122772 * np.cos(2 * np.pi * N / 366 + 1.498311) \
                              - 0.165458 * np.cos(4 * np.pi * N / 366 - 1.261546) \
                              - 0.005354 * np.cos(6 * np.pi * N / 366 - 1.1571)

# 太陽の時角 T_d_t [deg]（正午=0°、午前は負、午後は正）
T_d_t   = lambda H, ed, L       : (H  + ed - 12.0) * 15.0 + (L - 135.0)

# 角度[deg]の正弦・余弦
sin     = lambda v              : np.sin(np.radians(v))
cos     = lambda v              : np.cos(np.radians(v))

# 太陽高度 hs の正弦: sin(hs) = sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(時角)
sin_hs  = lambda L, dd, tdt     : sin(L) * sin(dd) + cos(L) * cos(dd) * cos(tdt)
# 太陽方位角 AZs の正弦/余弦（象限判定は arctan2 で別途実施）
sin_AZs = lambda dd, tdt, c_h   : cos(dd) * sin(tdt) / c_h
cos_AZs = lambda s_h, L, dd, c_h: (s_h * sin(L) - sin(dd)) / (c_h * cos(L))

def sun_loc(idx, lat = 36.00, lon = 140.00, td = -0.5):
    """太陽位置を簡易式で算出（赤緯/均時差/時角から）"""
    df = pd.DataFrame(index=idx)
    # 元日からの通し日数（正午寄せ）
    df['元日からの通し日数 N'] = [
        (i - datetime.datetime(i.year, 1, 1)).days + 1.5
        for i in idx
    ]
    # 小数時間（時＋分/60）に微調整 td を加える
    df['時刻 H'] = (
        idx.strftime("%H").astype('float64')
        + idx.strftime("%M").astype('float64') / 60
        + td
    )
    # 太陽の基本角
    df['太陽の赤緯 delta_d'] = delta_d(df['元日からの通し日数 N'])
    df['太陽の均時差 e_d']   = e_d(df['元日からの通し日数 N'])
    df['太陽の時角 T_d_t']   = T_d_t(df['時刻 H'], df['太陽の均時差 e_d'], lon)
    # 太陽高度
    df['太陽高度の正弦 sin_hs'] = sin_hs(
        lat, df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t']
    )
    df['太陽高度の余弦 cos_hs'] = np.sqrt(
        1 - np.power(df['太陽高度の正弦 sin_hs'], 2)
    )
    df['太陽高度 hs'] = np.degrees(
        np.arcsin(df['太陽高度の正弦 sin_hs'])
    )

    # 太陽方位角
    df['太陽方位角の正弦 sin_AZs'] = sin_AZs(
        df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t'], df['太陽高度の余弦 cos_hs']
    )
    df['太陽方位角の余弦 cos_AZs'] = cos_AZs(
        df['太陽高度の正弦 sin_hs'], lat, df['太陽の時角 T_d_t'], df['太陽高度の余弦 cos_hs']
    )

    df['太陽方位角 AZs'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs']
    )

    return df

def astro_sun_loc(idx, lat = '36 00 00.00', lon = '140 00 00.00', td = -0.5):
    """astropy を用いた太陽位置の高精度計算"""
    import astropy.time
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.coordinates import AltAz
    from astropy.coordinates import EarthLocation

    loc = EarthLocation(lat = lat, lon = lon)
    time = astropy.time.Time(idx) + (-9 + td ) * u.hour
    sun = get_sun(time).transform_to(AltAz(obstime = time, location = loc))

    df = pd.DataFrame(index = idx)

    # 太陽高度（仰角）
    df['太陽高度の正弦 sin_alt'] = np.array([np.sin(s.alt) for s in sun]).astype('float64')
    df['太陽高度の余弦 cos_alt'] = np.array([np.cos(s.alt) for s in sun]).astype('float64')
    df['太陽高度 alt'] = np.degrees(
        np.arcsin(df['太陽高度の正弦 sin_alt'])
    )

    # 太陽方位角
    df['太陽方位角の正弦 sin_az'] = np.array([np.sin(s.az) for s in sun]).astype('float64')
    df['太陽方位角の余弦 cos_az'] = np.array([np.cos(s.az) for s in sun]).astype('float64')

    df['太陽方位角 az'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_az'], df['太陽方位角の余弦 cos_az']
    )

    return df

def sep_direct_diffuse(s_ig, s_hs):
    """全天日射量と太陽高度から直散分離（Erbs）を行い Kt/Id/Ib を返す"""
    df = pd.concat([s_ig, s_hs], axis = 1)
    df.columns = ['水平面全天日射量', '太陽高度']
    df['晴天指数 Kt'] = Kt(Wh_to_MJ(df['水平面全天日射量']), df['太陽高度'])
    df['水平面拡散日射量 Id'] = MJ_to_Wh(Id(Wh_to_MJ(df['水平面全天日射量']), df['Kt']))
    df['法線面直達日射量 Ib'] = MJ_to_Wh(Ib(Wh_to_MJ(df['水平面全天日射量']), Wh_to_MJ(df['Id']), df['太陽高度']))
    return df

def eta(c):
    """角度補正の近似多項式（Horner 形）"""
    c2 = c * c
    return c * (2.3920 + c2 * (-3.8636 + c2 * (3.7568 + c2 * (-1.3968))))

def direc_solar(s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs):
    """直達・拡散・反射から方位別日射量（壁/ガラス/水平）を算出する"""
    df = pd.concat([s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs], axis = 1)
    df.columns = ['法線面直達日射量 Ib', '水平面拡散日射量 Id',
                  '太陽高度の正弦 sin_hs', '太陽高度の余弦 cos_hs', '太陽高度 hs',
                  '太陽方位角の正弦 sin_AZs', '太陽方位角の余弦 cos_AZs', '太陽方位角 AZs']

    df.loc[(df['太陽高度 hs'] > 0) & (-180 < df['太陽方位角 AZs']) & (df['太陽方位角 AZs'] < 0),   '直達日射量の東面成分 Ib_E'] = -1 * df['法線面直達日射量'] * df['太陽高度の余弦'] * df['太陽方位角の正弦']
    df.loc[(df['太陽高度 hs'] > 0) & (-90  < df['太陽方位角 AZs']) & (df['太陽方位角 AZs'] < 90),  '直達日射量の南面成分 Ib_S'] =      df['法線面直達日射量'] * df['太陽高度の余弦'] * df['太陽方位角の余弦']
    df.loc[(df['太陽高度 hs'] > 0) & (0    < df['太陽方位角 AZs']) & (df['太陽方位角 AZs'] < 180), '直達日射量の西面成分 Ib_W'] =      df['法線面直達日射量'] * df['太陽高度の余弦'] * df['太陽方位角の正弦']
    df.loc[(df['太陽高度 hs'] > 0) & (-180 < df['太陽方位角 AZs']) & (df['太陽方位角 AZs'] < -90), '直達日射量の北面成分 Ib_N'] = -1 * df['法線面直達日射量'] * df['太陽高度の余弦'] * df['太陽方位角の余弦']
    df.loc[(df['太陽高度 hs'] > 0) & (  90 < df['太陽方位角 AZs']) & (df['太陽方位角 AZs'] < 180), '直達日射量の北面成分 Ib_N'] = -1 * df['法線面直達日射量'] * df['太陽高度の余弦'] * df['太陽方位角の余弦']
    df.loc[df['太陽高度 hs'] > 0, '直達日射量の水平面成分 Ib_H'] = df['法線面直達日射量 Ib'] * df['太陽高度の正弦 sin_hs']
    df.loc[df['太陽高度 hs'] > 0, '水平面拡散日射量の反射成分 Id_R'] = (df['水平面拡散日射量 Id'] + df['法線面直達日射量 Ib']) * df['太陽高度の正弦 sin_hs'] * 0.5 * 0.1
    df['水平面拡散日射量の拡散成分 Id_D'] = df['水平面拡散日射量 Id'] * 0.5

    df = df.fillna(0)

    df['直達日射量の東面成分（ガラス） Ib_E_g']      = df['直達日射量の東面成分 Ib_E'] * eta(-1 * df['太陽高度の余弦 cos_hs'] * df['太陽方位角の正弦 sin_AZs'])
    df['直達日射量の南面成分（ガラス） Ib_S_g']      = df['直達日射量の南面成分 Ib_S'] * eta(     df['太陽高度の余弦 cos_hs'] * df['太陽方位角の余弦 cos_AZs'])
    df['直達日射量の西面成分（ガラス） Ib_W_g']      = df['直達日射量の西面成分 Ib_W'] * eta(     df['太陽高度の余弦 cos_hs'] * df['太陽方位角の正弦 sin_AZs'])
    df['直達日射量の北面成分（ガラス） Ib_N_g']      = df['直達日射量の北面成分 Ib_N'] * eta(-1 * df['太陽高度の余弦 cos_hs'] * df['太陽方位角の余弦 cos_AZs'])
    df['直達日射量の水平面成分（ガラス） Ib_H_g']    = df['直達日射量の水平面成分 Ib_H'] * eta(df['太陽高度の正弦 sin_hs'])
    df['水平面拡散日射量の反射成分（ガラス） Id_R_g'] = df['水平面拡散日射量の反射成分 Id_R'] * 0.808  # ガラス透過率 0.808
    df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] = df['水平面拡散日射量の拡散成分 Id_D'] * 0.808  # ガラス透過率 0.808

    df['日射熱取得量（東面）'] = df['直達日射量の東面成分 Ib_E']   + df['水平面拡散日射量の拡散成分 Id_D']   + df['水平面拡散日射量の反射成分 Id_R']
    df['日射熱取得量（南面）'] = df['直達日射量の南面成分 Ib_S']   + df['水平面拡散日射量の拡散成分 Id_D']   + df['水平面拡散日射量の反射成分 Id_R']
    df['日射熱取得量（西面）'] = df['直達日射量の西面成分 Ib_W']   + df['水平面拡散日射量の拡散成分 Id_D']   + df['水平面拡散日射量の反射成分 Id_R']
    df['日射熱取得量（北面）'] = df['直達日射量の北面成分 Ib_N']   + df['水平面拡散日射量の拡散成分 Id_D']   + df['水平面拡散日射量の反射成分 Id_R']
    df['日射熱取得量（水平面）'] = df['直達日射量の水平面成分 Ib_H']   + df['水平面拡散日射量の拡散成分 Id_D']   + df['水平面拡散日射量の反射成分 Id_R']
    df['日射熱取得量（東面ガラス）'] = df['直達日射量の東面成分（ガラス） Ib_E_g'] + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    df['日射熱取得量（南面ガラス）'] = df['直達日射量の南面成分（ガラス） Ib_S_g'] + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    df['日射熱取得量（西面ガラス）'] = df['直達日射量の西面成分（ガラス） Ib_W_g'] + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    df['日射熱取得量（北面ガラス）'] = df['直達日射量の北面成分（ガラス） Ib_N_g'] + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    df['日射熱取得量（水平面ガラス）'] = df['直達日射量の水平面成分（ガラス） Ib_H_g'] + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    return df

def make_solar(**kwargs):
    """方位別日射量の総合算出"""
    lat = kwargs['緯度'] if '緯度' in kwargs else 36.00
    lon = kwargs['経度'] if '経度' in kwargs else 140.00

    if '全天日射量' in kwargs:
        s_ig = kwargs['全天日射量']
        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            td = (s_ig.index[1] - s_ig.index[0]).seconds + (s_ig.index[1] - s_ig.index[0]).microseconds / 1000000 / 2 / 3600
        df = pd.concat([s_ig, sun_loc(s_ig.index, lat = lat, lon = lon, td = td)], axis = 1)
        df = pd.concat([df, sep_direct_diffuse(s_ig, df['hs'])], axis = 1)
        df = direc_solar(df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],
                         df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
                         df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs'])
    else:
        if '法線面直達日射量' in kwargs:    s_ib = kwargs['法線面直達日射量']
        else:                               raise Exception('ERROR: 法線面直達日射量 s_ib がありません。')
        if '水平面拡散日射量' in kwargs:    s_id = kwargs['水平面拡散日射量']
        else:                               raise Exception('ERROR: 水平面拡散日射量 s_id がありません。')
        if '時刻調整' in kwargs:
            td = kwargs['td']
        else:
            td = - (s_ib.index[1] - s_ib.index[0]).seconds + (s_ib.index[1] - s_ib.index[0]).microseconds / 1000000 / 2 / 3600
        df = pd.concat([s_ib, s_id, sun_loc(s_ib.index, lat = lat, lon = lon, td = td)], axis = 1)
        df = direc_solar(df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],
                         df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
                         df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs'])
    return df

__all__ = [
    "Kt", "Id", "Ib",
    "delta_d", "e_d", "T_d_t",
    "sun_loc", "astro_sun_loc",
    "sep_direct_diffuse", "eta", "direc_solar", "make_solar",
]



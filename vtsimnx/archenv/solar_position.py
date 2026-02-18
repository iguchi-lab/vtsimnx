import datetime

import numpy as np
import pandas as pd

from .archenv import _az_deg_from_sin_cos, _alt_deg_from_sin


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
    # 数値誤差で sin_hs が [-1, 1] を僅かに逸脱すると sqrt が NaN になったり、
    # cos_hs ≈ 0（天頂近傍）で方位角の sin/cos 計算が不安定になる。
    # ここでクリップ＆安全な cos_hs を作る。
    df['太陽高度の正弦 sin_hs'] = np.clip(df['太陽高度の正弦 sin_hs'], -1.0, 1.0)
    df['太陽高度の余弦 cos_hs'] = np.sqrt(
        np.clip(1 - np.power(df['太陽高度の正弦 sin_hs'], 2), 0.0, 1.0)
    )
    df['太陽高度 hs'] = _alt_deg_from_sin(df['太陽高度の正弦 sin_hs'])

    # 太陽方位角
    # cos_hs が極小だと分母が不安定になるので、近傍は方位角が定義できない前提で丸め込む
    eps = 1e-12
    safe_cos_hs = np.where(df['太陽高度の余弦 cos_hs'] < eps, np.nan, df['太陽高度の余弦 cos_hs'])
    df['太陽方位角の正弦 sin_AZs'] = sin_AZs(
        df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t'], safe_cos_hs
    )
    df['太陽方位角の余弦 cos_AZs'] = cos_AZs(
        df['太陽高度の正弦 sin_hs'], lat, df['太陽の赤緯 delta_d'], safe_cos_hs
    )
    # 数値誤差で [-1, 1] を僅かに逸脱することがあるためクリップ
    df['太陽方位角の正弦 sin_AZs'] = np.clip(df['太陽方位角の正弦 sin_AZs'], -1.0, 1.0)
    df['太陽方位角の余弦 cos_AZs'] = np.clip(df['太陽方位角の余弦 cos_AZs'], -1.0, 1.0)

    df['太陽方位角 AZs'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs']
    )

    return df


def astro_sun_loc(idx, lat = '36 00 00.00', lon = '140 00 00.00', td = -0.5):
    """astropy を用いた太陽位置の高精度計算
    lat, lon は DMS 表記の文字列を想定（例: '36 00 00.00'）
    戻り値: 仰角/方位角の sin, cos と角度 [deg]
    """
    # IERS（地球回転パラメータ）を自動取得して精度劣化の警告を避ける
    # オフライン等で取得できない場合もあるため、例外は握りつぶして続行する
    try:
        from astropy.utils import iers

        iers.conf.auto_download = True
        iers.conf.iers_auto_url = "https://datacenter.iers.org/data/9/finals2000A.all"
    except Exception:
        pass

    import astropy.time
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.coordinates import AltAz
    from astropy.coordinates import EarthLocation

    # 互換: float（deg）でも渡せるようにする
    # EarthLocation は unit 付きの文字列（例: "36d"）を受け付ける
    if isinstance(lat, (int, float)):
        lat = f"{float(lat)}d"
    if isinstance(lon, (int, float)):
        lon = f"{float(lon)}d"

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


__all__ = ["sun_loc", "astro_sun_loc"]


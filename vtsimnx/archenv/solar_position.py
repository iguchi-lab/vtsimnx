import warnings

import numpy as np
import pandas as pd

from .archenv import _az_deg_from_sin_cos, _alt_deg_from_sin


_COS_HS_EPS = 1e-12


# 太陽位置の計算に用いる基本式
# 太陽の赤緯 δ [deg]（年周の余弦和による近似）
def delta_d(N):
    """太陽の赤緯 δ [deg]。"""
    return (180 / np.pi) * (
        0.006322
        - 0.405748 * np.cos(2 * np.pi * N / 366 + 0.153231)
        - 0.005880 * np.cos(4 * np.pi * N / 366 + 0.207099)
        - 0.003233 * np.cos(6 * np.pi * N / 366 + 0.620129)
    )

# 太陽の均時差 ed [h]（平均太陽時と真太陽時の差）
def e_d(N):
    """太陽の均時差 e_d [h]。"""
    return (
        -0.000279
        + 0.122772 * np.cos(2 * np.pi * N / 366 + 1.498311)
        - 0.165458 * np.cos(4 * np.pi * N / 366 - 1.261546)
        - 0.005354 * np.cos(6 * np.pi * N / 366 - 1.1571)
    )

# 太陽の時角 T_d_t [deg]（正午=0°、午前は負、午後は正）
def T_d_t(H, ed, L):
    """太陽の時角 T_d_t [deg]（正午=0°）。"""
    return (H + ed - 12.0) * 15.0 + (L - 135.0)

# 角度[deg]の正弦・余弦
def sin_deg(v):
    """角度[deg]の正弦。"""
    return np.sin(np.radians(v))


def cos_deg(v):
    """角度[deg]の余弦。"""
    return np.cos(np.radians(v))

# 太陽高度 hs の正弦: sin(hs) = sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(時角)
def sin_hs(L, dd, tdt):
    """太陽高度 hs の正弦。"""
    return sin_deg(L) * sin_deg(dd) + cos_deg(L) * cos_deg(dd) * cos_deg(tdt)


# 太陽方位角 AZs の正弦/余弦（象限判定は arctan2 で別途実施）
def sin_AZs(dd, tdt, c_h):
    """太陽方位角 AZs の正弦。"""
    return cos_deg(dd) * sin_deg(tdt) / c_h


def cos_AZs(s_h, L, dd, c_h):
    """太陽方位角 AZs の余弦。"""
    return (s_h * sin_deg(L) - sin_deg(dd)) / (c_h * cos_deg(L))


# Backward-compatible aliases (internal use only)
sin = sin_deg
cos = cos_deg


def _build_time_columns(idx: pd.DatetimeIndex, td: float) -> tuple[pd.Series, pd.Series]:
    """sun_loc 用の N（日通し）と H（小数時）を構築する。"""
    n = pd.Series(idx.dayofyear.astype("float64") + 0.5, index=idx, name="元日からの通し日数 N")
    h = pd.Series(idx.hour.astype("float64") + idx.minute.astype("float64") / 60.0 + float(td), index=idx, name="時刻 H")
    return n, h


def sun_loc(idx, lat = 36.00, lon = 140.00, td = -0.5):
    """太陽位置を簡易式で算出（赤緯/均時差/時角から）
    idx: DatetimeIndex
    lat, lon: 緯度・経度 [deg]
    td: ローカル時刻微調整 [h]
    """
    df = pd.DataFrame(index=idx)
    n, h = _build_time_columns(idx, td)
    df["元日からの通し日数 N"] = n
    df["時刻 H"] = h
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
    safe_cos_hs = np.where(df['太陽高度の余弦 cos_hs'] < _COS_HS_EPS, np.nan, df['太陽高度の余弦 cos_hs'])
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
    except (ImportError, AttributeError) as e:
        warnings.warn(f"IERS auto configuration was skipped: {type(e).__name__}: {e}", RuntimeWarning)

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

    df = pd.DataFrame(index=idx)

    # 太陽高度（仰角）: hs
    sin_alt = np.array([np.sin(s.alt) for s in sun], dtype="float64")
    cos_alt = np.array([np.cos(s.alt) for s in sun], dtype="float64")
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    cos_alt = np.clip(cos_alt, -1.0, 1.0)
    hs = _alt_deg_from_sin(sin_alt)

    # 太陽方位角: astropy は North=0, East=90 の az を返す。
    # 本コードの AZs は South=0, East=-90, West=+90, North=±180 に合わせる。
    sin_az = np.array([np.sin(s.az) for s in sun], dtype="float64")
    cos_az = np.array([np.cos(s.az) for s in sun], dtype="float64")
    az = _az_deg_from_sin_cos(sin_az, cos_az)  # [deg], North=0
    azs = ((az - 180.0 + 180.0) % 360.0) - 180.0

    df["太陽高度の正弦 sin_hs"] = sin_alt
    df["太陽高度の余弦 cos_hs"] = cos_alt
    df["太陽高度 hs"] = hs
    df["太陽方位角 AZs"] = azs
    df["太陽方位角の正弦 sin_AZs"] = np.sin(np.radians(azs))
    df["太陽方位角の余弦 cos_AZs"] = np.cos(np.radians(azs))
    return df


__all__ = ["sun_loc", "astro_sun_loc"]


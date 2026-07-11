"""太陽位置（簡易式 / astropy）。列名は英語キー（``columns``）。"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from . import columns as C
from .archenv import _alt_deg_from_sin, _az_deg_from_sin_cos

_COS_HS_EPS = 1e-12


def delta_d(N: Any) -> Any:
    """太陽の赤緯 δ [deg]。"""
    return (180 / np.pi) * (
        0.006322
        - 0.405748 * np.cos(2 * np.pi * N / 366 + 0.153231)
        - 0.005880 * np.cos(4 * np.pi * N / 366 + 0.207099)
        - 0.003233 * np.cos(6 * np.pi * N / 366 + 0.620129)
    )


def e_d(N: Any) -> Any:
    """太陽の均時差 e_d [h]。"""
    return (
        -0.000279
        + 0.122772 * np.cos(2 * np.pi * N / 366 + 1.498311)
        - 0.165458 * np.cos(4 * np.pi * N / 366 - 1.261546)
        - 0.005354 * np.cos(6 * np.pi * N / 366 - 1.1571)
    )


def T_d_t(H: Any, ed: Any, L: float) -> Any:
    """太陽の時角 T_d_t [deg]（正午=0°）。"""
    return (H + ed - 12.0) * 15.0 + (L - 135.0)


def sin_deg(v: Any) -> Any:
    return np.sin(np.radians(v))


def cos_deg(v: Any) -> Any:
    return np.cos(np.radians(v))


def sin_hs(L: float, dd: Any, tdt: Any) -> Any:
    return sin_deg(L) * sin_deg(dd) + cos_deg(L) * cos_deg(dd) * cos_deg(tdt)


def sin_AZs(dd: Any, tdt: Any, c_h: Any) -> Any:
    return cos_deg(dd) * sin_deg(tdt) / c_h


def cos_AZs(s_h: Any, L: float, dd: Any, c_h: Any) -> Any:
    return (s_h * sin_deg(L) - sin_deg(dd)) / (c_h * cos_deg(L))


sin = sin_deg
cos = cos_deg


def _build_time_columns(idx: pd.DatetimeIndex, td: float) -> tuple[pd.Series, pd.Series]:
    n = pd.Series(idx.dayofyear.astype("float64") + 0.5, index=idx, name=C.DAY_OF_YEAR_N)
    h = pd.Series(
        idx.hour.astype("float64") + idx.minute.astype("float64") / 60.0 + float(td),
        index=idx,
        name=C.HOUR_H,
    )
    return n, h


def sun_loc(
    idx: pd.DatetimeIndex,
    lat: float = 36.00,
    lon: float = 140.00,
    td: float = -0.5,
) -> pd.DataFrame:
    """太陽位置を簡易式で算出。戻り列は英語キー。"""
    df = pd.DataFrame(index=idx)
    n, h = _build_time_columns(idx, td)
    df[C.DAY_OF_YEAR_N] = n
    df[C.HOUR_H] = h
    df[C.SOLAR_DECLINATION_DEG] = delta_d(df[C.DAY_OF_YEAR_N])
    df[C.EQUATION_OF_TIME_H] = e_d(df[C.DAY_OF_YEAR_N])
    df[C.HOUR_ANGLE_DEG] = T_d_t(df[C.HOUR_H], df[C.EQUATION_OF_TIME_H], lon)

    df[C.SIN_SOLAR_ALTITUDE] = np.clip(
        sin_hs(lat, df[C.SOLAR_DECLINATION_DEG], df[C.HOUR_ANGLE_DEG]), -1.0, 1.0
    )
    df[C.COS_SOLAR_ALTITUDE] = np.sqrt(
        np.clip(1 - np.power(df[C.SIN_SOLAR_ALTITUDE], 2), 0.0, 1.0)
    )
    df[C.SOLAR_ALTITUDE_DEG] = _alt_deg_from_sin(df[C.SIN_SOLAR_ALTITUDE])

    safe_cos_hs = np.where(df[C.COS_SOLAR_ALTITUDE] < _COS_HS_EPS, np.nan, df[C.COS_SOLAR_ALTITUDE])
    df[C.SIN_SOLAR_AZIMUTH] = np.clip(
        sin_AZs(df[C.SOLAR_DECLINATION_DEG], df[C.HOUR_ANGLE_DEG], safe_cos_hs), -1.0, 1.0
    )
    df[C.COS_SOLAR_AZIMUTH] = np.clip(
        cos_AZs(df[C.SIN_SOLAR_ALTITUDE], lat, df[C.SOLAR_DECLINATION_DEG], safe_cos_hs),
        -1.0,
        1.0,
    )
    df[C.SOLAR_AZIMUTH_DEG] = _az_deg_from_sin_cos(df[C.SIN_SOLAR_AZIMUTH], df[C.COS_SOLAR_AZIMUTH])
    return df


def astro_sun_loc(
    idx: pd.DatetimeIndex,
    lat: float | str = "36 00 00.00",
    lon: float | str = "140 00 00.00",
    td: float = -0.5,
) -> pd.DataFrame:
    """astropy を用いた太陽位置。戻り列は英語キー。"""
    try:
        from astropy.utils import iers

        iers.conf.auto_download = True
        iers.conf.iers_auto_url = "https://datacenter.iers.org/data/9/finals2000A.all"
    except (ImportError, AttributeError) as e:
        warnings.warn(f"IERS auto configuration was skipped: {type(e).__name__}: {e}", RuntimeWarning)

    import astropy.time
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun

    if isinstance(lat, (int, float)):
        lat = f"{float(lat)}d"
    if isinstance(lon, (int, float)):
        lon = f"{float(lon)}d"

    loc = EarthLocation(lat=lat, lon=lon)
    time = astropy.time.Time(idx) + (-9 + td) * u.hour
    sun = get_sun(time).transform_to(AltAz(obstime=time, location=loc))

    df = pd.DataFrame(index=idx)
    sin_alt = np.clip(np.array([np.sin(s.alt) for s in sun], dtype="float64"), -1.0, 1.0)
    cos_alt = np.clip(np.array([np.cos(s.alt) for s in sun], dtype="float64"), -1.0, 1.0)
    hs = _alt_deg_from_sin(sin_alt)

    sin_az = np.array([np.sin(s.az) for s in sun], dtype="float64")
    cos_az = np.array([np.cos(s.az) for s in sun], dtype="float64")
    az = _az_deg_from_sin_cos(sin_az, cos_az)
    azs = ((az - 180.0 + 180.0) % 360.0) - 180.0

    df[C.SIN_SOLAR_ALTITUDE] = sin_alt
    df[C.COS_SOLAR_ALTITUDE] = cos_alt
    df[C.SOLAR_ALTITUDE_DEG] = hs
    df[C.SOLAR_AZIMUTH_DEG] = azs
    df[C.SIN_SOLAR_AZIMUTH] = np.sin(np.radians(azs))
    df[C.COS_SOLAR_AZIMUTH] = np.cos(np.radians(azs))
    return df


__all__ = ["sun_loc", "astro_sun_loc"]

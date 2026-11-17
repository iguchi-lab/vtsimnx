"""標準住宅の方位・庇に、呼び出し側の直達・拡散・夜間放射を載せる。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from vtsimnx.archenv.nocturnal import nocturnal_gain_by_angles
from vtsimnx.archenv.solar_gain import solar_gain_by_angles, solar_gain_by_angles_with_shade

from .geometry import (
    LAT_DEG,
    LON_DEG,
    NOCTURNAL_TILTS,
    ROOF_SOLAR,
    SHADE_POLYS,
    SHADED_WINDOWS,
    WALL_SOLAR,
)

_SHADE_ORIGIN = "window_top_center"


def compute_solar(
    dni: pd.Series,
    dhi: pd.Series,
    *,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    walls: Mapping[str, Mapping[str, Any]] | None = None,
    roofs: Mapping[str, Mapping[str, Any]] | None = None,
    windows: Mapping[str, Mapping[str, Any]] | None = None,
    shade_polys: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """壁・屋根・庇付き窓の日射列を全部返す。気象系列だけ呼び出し側。"""
    lat = LAT_DEG if lat_deg is None else lat_deg
    lon = LON_DEG if lon_deg is None else lon_deg
    wall_planes = WALL_SOLAR if walls is None else walls
    roof_planes = ROOF_SOLAR if roofs is None else roofs
    window_planes = SHADED_WINDOWS if windows is None else windows
    shades = SHADE_POLYS if shade_polys is None else shade_polys

    solar = pd.DataFrame(index=dni.index)
    for name, spec in wall_planes.items():
        solar[name] = solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat, lon_deg=lon, **spec)
    for name, spec in roof_planes.items():
        solar[name] = solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat, lon_deg=lon, **spec)
    for name, spec in window_planes.items():
        window = dict(spec)
        shade_key = window.pop("shade_key")
        solar[name] = solar_gain_by_angles_with_shade(
            dni=dni,
            dhi=dhi,
            lat_deg=lat,
            lon_deg=lon,
            shade_coords=shades[shade_key],
            shade_origin_mode=_SHADE_ORIGIN,
            glass=True,
            **window,
        )
    return solar


def compute_nocturnal(
    rn_horizontal: pd.Series,
    *,
    tilts: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """標準住宅の傾斜ごとの夜間放射列を返す。"""
    tilt_map = NOCTURNAL_TILTS if tilts is None else tilts
    nocturnal = pd.DataFrame(index=rn_horizontal.index)
    for name, tilt_deg in tilt_map.items():
        nocturnal[name] = nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=tilt_deg)
    return nocturnal

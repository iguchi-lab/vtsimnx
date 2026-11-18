"""標準住宅の方位・庇に、呼び出し側の直達・拡散・夜間放射を載せる。"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from vtsimnx.archenv.nocturnal import nocturnal_gain_by_angles
from vtsimnx.archenv.solar_gain import solar_gain_by_angles, solar_gain_by_angles_with_shade

from .geometry import (
    LAT_DEG,
    LD_SOUTH_SOLAR_KEY,
    LD_SOUTH_SPLIT,
    LD_SOUTH_SPLIT_SIZE,
    LON_DEG,
    NOCTURNAL_TILTS,
    ROOF_SOLAR,
    SHADE_POLYS,
    SHADED_WINDOWS,
    WALL_SOLAR,
)

_SHADE_ORIGIN = "window_top_center"


def _windows_for_shades(
    windows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """LD南は東西2窓で日除けが違う。部位は1面なので平均用の列名を返す。"""
    planes = {name: dict(spec) for name, spec in windows.items()}
    if LD_SOUTH_SOLAR_KEY not in planes:
        return planes, None
    combined = planes.pop(LD_SOUTH_SOLAR_KEY)
    for name, shade_key in LD_SOUTH_SPLIT:
        planes[name] = {**combined, "shade_key": shade_key, **LD_SOUTH_SPLIT_SIZE}
    return planes, LD_SOUTH_SOLAR_KEY


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
    window_planes, ld_south = _windows_for_shades(SHADED_WINDOWS if windows is None else windows)
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
    if ld_south is not None:
        west, east = (name for name, _key in LD_SOUTH_SPLIT)
        # LD南の既存表面は 6.93 m²。2窓が等面積・同仕様なので平均日射量を渡す。
        solar[ld_south] = (solar[west] + solar[east]) / 2
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

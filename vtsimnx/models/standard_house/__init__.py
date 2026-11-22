"""標準住宅の部品カタログ。

気象は組み込まない。室・層・庇・換気・空調などを名前で選び、利用者が組み立てる。

::

    import vtsimnx as vt
    h = vt.standard_house
    list(h.ventilation)          # ['24h']
    vent = h.ventilation["24h"]  # コピー。このあと枝を足したり消したりできる
    acs = h.aircon["ac4"]        # 4台。機種や台数を差し替えてよい
"""

from functools import partial
from typing import Any

from .catalog import PartCatalog, snapshot
from .envelope import (
    GEOMETRIC_VOLUME,
    M_CAPA_PER_M3,
    ROOM_VOLUME,
    SURFACE_ROWS,
    T_CAPA_PER_M3,
    U_VALUE_WINDOW,
    WHOLE_OPENING_U,
    bind_surface_types,
    compose_surfaces,
    face_to_face_u,
    make_layers,
    make_surface_types,
    room_nodes,
)
from .foundation import foundation_layers
from .gaps import GAP_CASES, central_aircon, facade_pressures, gap_network, gap_nodes
from .geometry import (
    CASE_ID,
    CASE_NAME,
    LAT_DEG,
    LON_DEG,
    WARMUP_START,
    NOCTURNAL_TILTS,
    ROOF_SOLAR,
    SHADE_POLYS,
    SHADED_WINDOWS,
    WALL_SOLAR,
)
from .grade_data import GRADE_PATHS
from .grades import grade_layers, grade_surface_rows, grade_surface_types
from .rating import rating_template
from .irradiance import compute_nocturnal, compute_solar
from .systems import (
    HEAT_RATIO,
    make_aircon_ac4,
    make_aircon_specs,
    make_heat_sources,
    make_humidity_sources,
    make_ventilation_24h,
)

rooms: PartCatalog[dict[str, float]] = PartCatalog({"default": lambda: snapshot(ROOM_VOLUME)})
layers: PartCatalog[dict[str, list[dict[str, Any]]]] = PartCatalog({
    "default": make_layers,
    **{f"grade{g}": partial(grade_layers, g) for g in GRADE_PATHS},
    **{f"foundation{g}": partial(foundation_layers, g) for g in range(4, 8)},
})
shade: PartCatalog[dict[str, Any]] = PartCatalog({"default": lambda: snapshot(SHADE_POLYS)})
solar_plane: PartCatalog[dict[str, Any]] = PartCatalog({
    "wall": lambda: snapshot(WALL_SOLAR),
    "roof": lambda: snapshot(ROOF_SOLAR),
    "window_shade": lambda: snapshot(SHADED_WINDOWS),
})
nocturnal_tilt: PartCatalog[dict[str, Any]] = PartCatalog({"default": lambda: snapshot(NOCTURNAL_TILTS)})
surface_types: PartCatalog[dict[str, dict[str, Any]]] = PartCatalog({
    "default": make_surface_types,
    **{f"grade{g}": partial(grade_surface_types, g) for g in GRADE_PATHS},
    "foundation4": make_surface_types,
    **{f"foundation{g}": partial(grade_surface_types, g) for g in GRADE_PATHS},
})
surface_rows: PartCatalog[list[Any]] = PartCatalog({
    "default": lambda: snapshot(SURFACE_ROWS),
    **{f"grade{g}": partial(grade_surface_rows, g) for g in GRADE_PATHS},
    "foundation4": lambda: snapshot(SURFACE_ROWS),
    **{f"foundation{g}": partial(grade_surface_rows, g) for g in GRADE_PATHS},
})
ventilation: PartCatalog[list[dict[str, Any]]] = PartCatalog({"24h": make_ventilation_24h})
aircon: PartCatalog[list[dict[str, Any]]] = PartCatalog({"ac4": make_aircon_ac4})
aircon_spec: PartCatalog[dict[str, Any]] = PartCatalog(
    {name: (lambda n=name: make_aircon_specs()[n]) for name in ("highspec", "standard", "rac", "rac_standard")}
)
heat_source: PartCatalog[list[dict[str, Any]]] = PartCatalog({"default": make_heat_sources})
humidity_source: PartCatalog[list[dict[str, Any]]] = PartCatalog({"default": make_humidity_sources})


def available() -> dict[str, list[str]]:
    """登録済み部品名。"""
    return {
        "rooms": rooms.names(),
        "layers": layers.names(),
        "shade": shade.names(),
        "solar_plane": solar_plane.names(),
        "nocturnal_tilt": nocturnal_tilt.names(),
        "surface_types": surface_types.names(),
        "surface_rows": surface_rows.names(),
        "ventilation": ventilation.names(),
        "aircon": aircon.names(),
        "aircon_spec": aircon_spec.names(),
        "heat_source": heat_source.names(),
        "humidity_source": humidity_source.names(),
    }


__all__ = [
    "rating_template",
    "GAP_CASES",
    "central_aircon",
    "facade_pressures",
    "gap_network",
    "gap_nodes",
    "CASE_ID",
    "CASE_NAME",
    "LAT_DEG",
    "LON_DEG",
    "WARMUP_START",
    "T_CAPA_PER_M3",
    "M_CAPA_PER_M3",
    "U_VALUE_WINDOW",
    "WHOLE_OPENING_U",
    "GEOMETRIC_VOLUME",
    "face_to_face_u",
    "HEAT_RATIO",
    "rooms",
    "layers",
    "shade",
    "solar_plane",
    "nocturnal_tilt",
    "surface_types",
    "surface_rows",
    "ventilation",
    "aircon",
    "aircon_spec",
    "heat_source",
    "humidity_source",
    "available",
    "room_nodes",
    "bind_surface_types",
    "compose_surfaces",
    "compute_solar",
    "compute_nocturnal",
]

"""部位の並列断面を保持した標準住宅の等級別外皮カタログ。"""

from .envelope import (
    SURFACE_ROWS,
    _ALPHA_I_CONV,
    _ALPHA_O_CONV,
    _ALPHA_I_FILM,
    _ALPHA_O_FILM,
    make_layers,
    make_surface_types,
)
from .grade_data import GRADE_PATHS, MATERIAL_PROPERTIES, OPENING_PROPERTIES, OPENING_ROWS, WALL_ROWS


def grade_layers(grade: int) -> dict:
    layers = make_layers()
    for part, paths in GRADE_PATHS[grade].items():
        for i, (_, stack) in enumerate(paths, 1):
            layers[f"{part}_断面{i}"] = [
                {"key": material, "t": mm / 1000, **MATERIAL_PROPERTIES[material]} for material, mm in stack
            ]
    return layers


def grade_surface_types(grade: int) -> dict:
    types = make_surface_types()
    for template, part in dict.fromkeys((row[1], row[3]) for row in WALL_ROWS):
        for i in range(1, len(GRADE_PATHS[grade][part]) + 1):
            layer = f"{part}_断面{i}"
            types[f"{template}_{layer}"] = dict(types[template], layer_key=layer)
    for template, (u, scr, scc) in OPENING_PROPERTIES[grade].items():
        # 部位表のU値は室内外の表面熱抵抗込み。solverには面間の値を渡し、alphaは対流のみ。
        types[template]["u_value"] = 1 / (1 / u - 1 / _ALPHA_O_FILM - 1 / _ALPHA_I_FILM)
        types[template]["alpha_i"] = _ALPHA_I_CONV
        types[template]["alpha_o"] = _ALPHA_O_CONV
        if scr is not None:
            types[template].update(SCR=scr, SCC=scc)
    return types


def grade_surface_rows(grade: int) -> list[dict]:
    rows = []
    for row in SURFACE_ROWS:
        key = row["key"]
        attic_exterior = key.startswith(("切妻->外部", "片流れ(3)->外部", "片流れ(4)->外部"))
        if not attic_exterior and key.endswith(("_一般部", "_熱橋部", "外皮天井")):
            continue
        if row["surface"] not in OPENING_PROPERTIES[grade]:
            rows.append(row.copy())
    for key, template, area, part in WALL_ROWS:
        for i, (percent, _) in enumerate(GRADE_PATHS[grade][part], 1):
            rows.append(
                {
                    "key": f"{key}_断面{i}",
                    "surface": f"{template}_{part}_断面{i}",
                    "area": area * percent / 100,
                }
            )
    for key, template, area in OPENING_ROWS:
        rows.append({"key": key, "surface": template, "area": area})
    return rows

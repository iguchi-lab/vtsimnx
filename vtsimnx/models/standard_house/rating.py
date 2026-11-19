"""標準住宅の床・基礎断熱境界を省エネ計算用の部位に対応づける。"""

import re

from .grade_data import OPENING_PROPERTIES


def rating_template(model="default", *, region=6):
    """評価条件の編集用辞書。開口部性能と土間外周の設計値を記入して使用する。"""
    from . import surface_rows, surface_types

    foundation = model.startswith("foundation")
    rows = surface_rows[model]
    opening_data = OPENING_PROPERTIES.get(int(model[-1]), {}) if model != "default" else {}
    types = surface_types[model]
    outside_nodes = {"外部", "地下1m", "切妻", "片流れ(3)", "片流れ(4)"}
    if not foundation:
        outside_nodes.add("床下")
    parts = {}
    excluded = []
    for row in rows:
        key = row["key"]
        first, second = key.split("||")[0].split("->")
        if (first in outside_nodes) == (second in outside_nodes):
            excluded.append(key)
            continue
        adjacent = first if first in outside_nodes else second
        template = row["surface"]
        spec = types[template]
        # 入力の一般部・熱橋部は同じ面の並列断面。面積の端数処理は合算後。
        group = re.sub(r"_(?:一般部|熱橋部|断面\d+)$", "", key)
        if group in parts:
            parts[group]["keys"].append(key)
            continue
        item = {"keys": [key], "outside": adjacent == "外部", "H": 0.7 if adjacent == "床下" else 1, "directions": []}
        if adjacent == "地下1m":
            item["kind"] = "slab"
        elif spec["part"] == "glass":
            item.update(kind="window", U=None, eta_d=None, frame_type=None, glass_type=None, attachment=None)
            if model in {"default", "foundation4"}:
                # 等級4 UA.xlsx の窓仕様。基礎断熱ケースでも開口部は共通。
                item.update(frame_type="金属製など", glass_type="一般複層", attachment="なし", U=4.65)
        elif "ドア" in template:
            item.update(kind="door", U=None)
        elif "天井" in template:
            item["kind"] = "ceiling"
        elif template.startswith("外皮床"):
            item["kind"] = "floor"
        else:
            item["kind"] = "wall"
        if template in opening_data:
            item["U"] = opening_data[template][0]
        if adjacent == "外部":
            direction = template.split("_")[0]
            if direction not in {"E", "W", "S", "N"}:
                raise ValueError(f"{key}: missing facade direction")
            item["directions"] = [direction]
        parts[group] = item
    return {
        "region": region,
        "parts": list(parts.values()),
        "excluded": excluded,
        "lines": [
            {"name": "土間外周（接する空間ごとに分割）", "kind": "perimeter", "length": None, "psi": None, "H": 1}
        ],
    }

"""標準住宅の床・基礎断熱境界を省エネ計算用の部位に対応づける。"""

import re

from .grade_data import OPENING_PROPERTIES


def rating_template(model="default", *, region=6):
    """評価条件の編集用辞書。開口部性能と土間外周の設計値を記入して使用する。"""
    from . import surface_rows, surface_types

    foundation = model.startswith("foundation")
    rows = surface_rows[model]
    grade = 4 if model == "default" else int(model[-1])
    opening_data = OPENING_PROPERTIES.get(grade, {})
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
            if item["kind"] == "window":
                glass = {5: "一般複層", 6: "Low-E二層・取得型", 7: "Low-E三層・複数膜・遮蔽型"}[grade]
                # 各等級の UA.xlsx が指定する日射評価条件（和室も付属部材なし）。
                item.update(frame_type="木製・樹脂製", glass_type=glass, attachment="なし")
                if template == "S_和室_窓":
                    bare_u = {5: 3.49, 6: 2.33, 7: 1.60}[grade]
                    item["U"] = (bare_u + 1 / (1 / bare_u + 0.18)) / 2
        if adjacent == "外部":
            direction = template.split("_")[0]
            if direction not in {"E", "W", "S", "N"}:
                raise ValueError(f"{key}: missing facade direction")
            item["directions"] = [direction]
        parts[group] = item
    exterior = {
        "name": "外気側土間外周",
        "kind": "perimeter",
        "H": 1,
        "psi": None,
        "length": None,
        "wall_area": None,
        "wall_keys": [],
    }
    lines = [exterior]
    if foundation:
        # InputFace 1階床の外周。南北向きの辺各7.28 m、東西向きの辺各10.465 m。
        exterior["length"] = 2 * (10.465 + 7.28)
        exterior["wall_keys"] = [row["key"] for row in rows if row["surface"].endswith("_基礎外壁")]
    else:
        # 換気床下の外周は断熱境界外。玄関土間の外気側立上りの面積は独立入力。
        lines.append(
            {
                **exterior,
                "name": "玄関土間・床下側",
                "H": 0.7,
                "wall_keys": [row["key"] for row in rows if row["surface"] == "ホール-床下"],
            }
        )
    return {"region": region, "parts": list(parts.values()), "excluded": excluded, "lines": lines}

"""外壁一般部の断熱材と同じ熱抵抗をもつ基礎外断熱。"""

from .envelope import make_layers
from .grades import grade_layers

_INSULATION = {
    "住宅用グラスウール断熱材16K相当",
    "【H28基準】高性能グラスウール断熱材16K相当",
    "【H28基準】押出法ポリスチレンフォーム保温板3種",
}


def foundation_layers(grade: int) -> dict:
    """床の構造材を保持し、熱橋を除く外壁断熱材のΣ(t/λ)を基礎外側のXPSで再現する。"""
    layers = make_layers() if grade == 4 else grade_layers(grade)
    wall = layers["外壁_一般部" if grade == 4 else "外壁_断面1"]
    resistance = sum(item["t"] / item["lambda"] for item in wall if item["key"] in _INSULATION)
    xps = layers["ホール-床下"].pop()
    xps["t"] = resistance * xps["lambda"]
    layers["基礎外壁"].append(xps)
    for name, stack in layers.items():
        if name.startswith("外皮床_"):
            layers[name] = [item for item in stack if item["key"] not in _INSULATION]
    return layers

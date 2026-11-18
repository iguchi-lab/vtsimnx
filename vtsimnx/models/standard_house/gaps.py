"""共有Colabの外皮・建具の隙間、給排気口、全館空調ダクト。"""

from math import isfinite, pi
from typing import Any

from .envelope import ROOM_VOLUME, room_nodes

# 方位、室、幅、高さ、下端高さ [m]。LD南の2窓は独立した開口。
_WINDOWS = [
    ("南", "和室", 2.550, 1.800, 0.510),
    ("南", "LD", 1.650, 2.100, 0.510),
    ("南", "LD", 1.650, 2.100, 0.510),
    ("東", "LD", 1.650, 1.300, 1.010),
    ("東", "台所", 1.400, 0.700, 1.610),
    ("北", "台所", 0.900, 1.800, 0.510),
    ("西", "浴室", 0.600, 0.900, 1.410),
    ("北", "1階トイレ", 0.600, 0.900, 1.410),
    ("北", "洗面所", 0.600, 0.900, 1.410),
    ("北", "ホール", 0.600, 0.900, 1.410),
    ("西", "ホール", 0.900, 2.100, 0.000),
    ("西", "クローゼット", 0.600, 0.900, 4.310),
    ("南", "寝室", 1.650, 1.050, 4.310),
    ("西", "寝室", 0.900, 1.100, 4.310),
    ("南", "子供室1", 1.650, 1.950, 3.410),
    ("南", "子供室2", 1.650, 1.950, 3.410),
    ("東", "子供室2", 0.600, 1.100, 4.310),
    ("北", "2階ホール", 0.900, 1.100, 4.447),
    ("北", "2階トイレ", 0.600, 0.900, 4.447),
]

# 開き戸・片引き戸（幅0.79 m、高さ1.80 m）の接続と下端高さ。
_DOORS = [
    ("和室", "ホール", "和室-ホール", 0.510),
    ("LD", "ホール", "LDK-ホール_1", 0.510),
    ("LD", "ホール", "LDK-ホール_2", 0.510),
    ("LD", "ホール", "LDK-ホール_3", 0.510),
    ("浴室", "洗面所", "浴室-洗面所", 0.510),
    ("ホール", "1階トイレ", "ホール-1階トイレ", 0.510),
    ("ホール", "洗面所", "LDK-洗面所", 0.510),
    ("クローゼット", "寝室", "クローゼット-寝室", 3.410),
    ("2階ホール", "寝室", "2階ホール-寝室", 3.410),
    ("2階ホール", "子供室1", "2階ホール-子供室1", 3.410),
    ("2階ホール", "子供室2", "2階ホール-子供室2", 3.410),
    ("2階ホール", "2階トイレ", "2階ホール-2階トイレ", 3.410),
]

GAP_CASES = {0: 0.00, 0.5: 0.30, 1: 0.59, 2: 1.19, 4: 2.37, 8: 4.74}


def _gap(link, width, height, sill, coefficient, exponent, verticals):
    edges = [("下端", sill, width), ("上端", sill + height, width)]
    for i in range(10):
        edges.append((f"縦{i + 1}", sill + height * (i + 0.5) / 10, verticals * height / 10))
    return [
        {
            "key": f"{link} {edge}",
            "type": "gap",
            "a": coefficient * length / 3600,
            "n": exponent,
            "h_from": z,
            "h_to": z,
        }
        for edge, z, length in edges
    ]


def gap_network(multiplier: float = 1, *, central_air: bool = False) -> list[dict[str, Any]]:
    """基礎断熱住宅の換気枝。倍率は外皮隙間だけに掛ける。"""
    if not isfinite(multiplier) or multiplier < 0:
        raise ValueError("multiplier must be finite and nonnegative")
    register = {
        "type": "pressure_loss",
        "area": pi * 0.05**2,
        "friction_factor": 0.036,
        "length": 53.5,
        "diameter": 0.1,
    }
    branches: list[dict[str, Any]] = [
        {"key": key, **register}
        for key in ("南->和室", "北->台所", "南->LD", "北->クローゼット", "南->寝室", "南->子供室1", "南->子供室2")
    ]
    for room in ("1階トイレ", "2階トイレ"):
        branches.append({"key": f"{room}->北", "q_max": 90 / 3600, "q1": 75 / 3600, "p1": 30.0, "p_max": 110.0})
    for key in ("ホール->2階ホール", "LD->台所"):
        branches.append({"key": key, "alpha": 0.65, "area": 2.0})
    for room in ("切妻", "片流れ(3)", "片流れ(4)"):
        branches.append({"key": f"外部->{room}->外部", "vol": ROOM_VOLUME[room] * 5 / 3600})
    if multiplier:
        for i, (face, room, w, h, sill) in enumerate(_WINDOWS, 1):
            branches.extend(_gap(f"{face}->{room} || 開口{i}", w, h, sill, 0.2748 * multiplier, 1.81, 3))
    branches.extend(_gap("LD->和室 || LDK-和室", 1.8, 1.8, 0.51, 84.94 / (5 * 1.8), 1.56, 3))
    for source, target, door, sill in _DOORS:
        branches.extend(_gap(f"{source}->{target} || {door}", 0.79, 1.8, sill, 65.48 / (2 * (0.79 + 1.8)), 1.74, 2))
    if central_air:
        for i, (room, length) in enumerate(
            [
                ("LD", 25.6 / 3),
                ("LD", 25.6 / 3),
                ("台所", 25.6 / 3),
                ("和室", 8.6),
                ("寝室", 10.2),
                ("子供室1", 11.8),
                ("子供室2", 8.1),
            ],
            1,
        ):
            branches.append(
                {
                    "key": f"AC出口->{room} || 系統{i}",
                    "type": "pressure_loss",
                    "area": pi * 0.075**2,
                    "friction_factor": 0.02,
                    "length": length,
                    "diameter": 0.15,
                }
            )
    return branches


def gap_nodes(temperature, humidity, pressure, *, central_air=False):
    """室と外気節点。humidity は kg/kg(DA)、pressure は方位別 Pa。地下1mは別途追加する。"""
    nodes = room_nodes()
    rooms = {room for _, room, *_ in _WINDOWS}
    for node in nodes:
        if node["key"] in rooms:
            node["calc_p"] = True
    nodes.append({"key": "外部", "t": temperature, "x": humidity})
    for face in "東南西北":
        nodes.append({"key": face, "t": temperature, "x": humidity, "p": pressure[face]})
    if central_air:
        nodes.append({"key": "AC出口", "calc_p": True, "calc_t": True, "calc_x": True})
    return nodes


def facade_pressures(weather, coefficients):
    """Colabの気象表（外気温 °C、風速 m/s、風向1～16）と16方位の風圧係数表からPaを返す。"""
    density = 101324.6 / (287.05 * (273.15 + weather["外気温"]))
    dynamic = 0.5 * density * weather["風速"] ** 2
    table = coefficients.loc[list(range(1, 17))]
    return {face: (weather["風向"].map(table[face + "面"]).fillna(0) * dynamic).tolist() for face in "東南西北"}


def central_aircon():
    """Colabと同じ全館空調。gap_nodes/gap_network の central_air=True と組み合わせる。"""
    from vtsimnx import schedule

    ductcentral_ac = {
        "Q": {
            "cooling": {"min": 2.682626, "mid": 3.832323, "rtd": 7.664646},  # 冷房能力[kW]
            "heating": {"min": 2.706897, "mid": 3.866996, "rtd": 7.733992},
        },  # 暖房能力[kW]
        "P": {
            "cooling": {"rtd": 2.417870},  # 冷房時消費電力[kW]
            "heating": {"rtd": 2.056912},
        },  # 暖房時消費電力[kW]
        "P_fan": {
            "cooling": {"mid": 0.188513, "rtd": 0.240326},  # 冷房時送風ファン消費電力[kW]
            "heating": {"mid": 0.188981, "rtd": 0.241263},
        },  # 暖房時送風ファン消費電力[kW]
        "V_inner": {
            "cooling": {
                "mid": 1258.598 / 3600,
                "dsgn": 1301.284 / 3600,
                "rtd": 1647.195 / 3600,
            },  # 冷房時の室内機の送風量 [m3/s]
            "heating": {"mid": 1262.113 / 3600, "dsgn": 1306.839 / 3600, "rtd": 1654.227 / 3600},
        },  # 暖房時の室内機の送風量 [m3/s]
    }

    return [
        {
            "key": "全館空調",
            "set": "LD",
            "in": "2階ホール",
            "out": "AC出口",
            "outside": "外部",
            "mode": schedule.ac_mode["region6"]["全館空調"],
            "pre_temp": schedule.pre_tmp["region6"]["全館空調"],
            "model": "DUCT_CENTRAL",
            "ac_spec": ductcentral_ac,
            # 定格（λ=1）のファン曲線。圧力は Pa、風量は m³/s
            "p_max": 440.0,
            "p1": 200.0,
            "q1": 1360 / 3600,
            "q_max": 1880 / 3600,
            # 任意。空調→AC出口の短いつなぎ。省略時は area=0.05, k_total=1
            "area": 0.05,
            "k_total": 0.01,
        }
    ]

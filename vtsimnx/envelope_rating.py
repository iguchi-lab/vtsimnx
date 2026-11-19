"""住宅の外皮面積を用いる標準計算（第三章第二節 8）。"""

import warnings
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

from .window_rating import window_eta, window_u

D = Decimal

# 第三章第二節 付録C 表1・2。列は地域1から順。
_NU_H = {
    "N": (0.260, 0.263, 0.284, 0.256, 0.238, 0.261, 0.227),
    "NE": (0.333, 0.341, 0.348, 0.330, 0.310, 0.325, 0.281),
    "E": (0.564, 0.554, 0.540, 0.531, 0.568, 0.579, 0.543),
    "SE": (0.823, 0.766, 0.751, 0.724, 0.846, 0.833, 0.843),
    "S": (0.935, 0.856, 0.851, 0.815, 0.983, 0.936, 1.023),
    "SW": (0.790, 0.753, 0.750, 0.723, 0.815, 0.763, 0.848),
    "W": (0.535, 0.544, 0.542, 0.527, 0.538, 0.523, 0.548),
    "NW": (0.325, 0.341, 0.351, 0.326, 0.297, 0.317, 0.284),
    "Top": (1,) * 7,
    "Bottom": (0,) * 7,
}
_NU_C = {
    "N": (0.329, 0.341, 0.335, 0.322, 0.373, 0.341, 0.307, 0.325),
    "NE": (0.430, 0.412, 0.390, 0.426, 0.437, 0.431, 0.415, 0.414),
    "E": (0.545, 0.503, 0.468, 0.518, 0.500, 0.512, 0.509, 0.515),
    "SE": (0.560, 0.527, 0.487, 0.508, 0.500, 0.498, 0.490, 0.528),
    "S": (0.502, 0.507, 0.476, 0.437, 0.472, 0.434, 0.412, 0.480),
    "SW": (0.526, 0.548, 0.550, 0.481, 0.520, 0.491, 0.479, 0.517),
    "W": (0.508, 0.529, 0.553, 0.481, 0.518, 0.504, 0.495, 0.505),
    "NW": (0.411, 0.428, 0.447, 0.401, 0.442, 0.427, 0.406, 0.411),
    "Top": (1,) * 8,
    "Bottom": (0,) * 8,
}


def _number(value, name, *, positive=False, maximum=None):
    try:
        n = D(str(value))
    except ArithmeticError as exc:
        raise ValueError(f"{name}: numeric value required") from exc
    if not n.is_finite() or n < 0 or (positive and n == 0) or (maximum is not None and n > maximum):
        raise ValueError(f"{name}: invalid value {value!r}")
    return n


def _u(surface, spec):
    if spec["kind"] == "window" and spec.get("U") is None:
        if surface.get("whole_window_u") is not None:
            return _number(surface["whole_window_u"], "whole_window_u", positive=True)
        return D(str(window_u({**surface, **{k: v for k, v in spec.items() if v is not None}})))
    if "U" in spec:
        return _number(spec["U"], f"{surface['key']}: U", positive=True)
    # 入力順は室内側から屋外側。通気層に達したら、その外側の仕上げを除く。
    film = {"wall": ".11", "floor": ".15", "ceiling": ".09", "roof": ".09"}[spec["kind"]]
    exposed = spec["outside"] and not any(layer.get("ventilated_air_layer") for layer in (surface.get("layers") or []))
    resistance = D(film) + (D(".04") if exposed else D(film))
    if not surface.get("layers"):
        raise ValueError(f"{surface['key']}: specify whole-component U or layers")
    for layer in surface["layers"]:
        if layer.get("ventilated_air_layer"):
            break
        if "thermal_resistance" in layer:
            resistance += _number(layer["thermal_resistance"], "thermal_resistance")
        else:
            resistance += _number(layer["t"], "t", positive=True) / _number(layer["lambda"], "lambda", positive=True)
    return 1 / resistance


def _solar(spec, season, region, conductivity):
    table = _NU_C if season == "C" else _NU_H
    directions = spec["directions"]
    if not directions:
        return D(0)
    nu = sum(D(str(table[d][region - 1])) for d in directions if d is not None) / D(len(directions))
    if spec["kind"] == "window":
        coefficient = spec.get("eta_d_" + season, spec.get("eta_d"))
        eta = _number(coefficient, f"{spec['keys'][0]}: eta_d", maximum=1) * _number(
            spec.get("f_" + season, 0.93 if season == "C" else 0.51), "f_" + season, maximum=1
        )
    else:
        eta = D(".034") * conductivity * _number(spec.get("gamma_" + season, 1), "gamma", maximum=1)
        eta *= _number(spec.get("absorptance", 0.8), "absorptance", maximum=1) / D(".8")
    return eta * nu


def _line_length(line, surfaces, used_keys):
    """面積は当該区間の立上り見付面積。土間の水平面積は使わない。"""
    if line.get("length") is not None or line["kind"] != "perimeter":
        length = _number(line.get("length"), f"{line['name']}: length", positive=True)
        return length, {"length": float(length), "length_source": "input"}
    keys = line.get("wall_keys") or []
    if line.get("wall_area") is not None:
        area = _number(line["wall_area"], "wall_area", positive=True)
        source = "wall_area"
    elif keys:
        area = D(0)
        for key in keys:
            if key not in surfaces or key in used_keys:
                raise ValueError(f"{line['name']}: unknown or repeated foundation wall {key}")
            if surfaces[key].get("part") not in (None, "wall"):
                raise ValueError(f"{key}: wall_keys must reference vertical foundation walls")
            used_keys.add(key)
            area += _number(surfaces[key]["area"], key, positive=True)
        source = "wall_keys"
    else:
        raise ValueError(f"{line['name']}: specify length [m], wall_area [m²] or wall_keys")
    length = area / D("0.4")
    warnings.warn(f"{line['name']}: 外周長が未入力。立上り面積 {area} m² / 0.4 m = {length} m を採用", stacklevel=3)
    return length, {"length": float(length), "length_source": source, "wall_area": float(area)}


def envelope_performance(input_data, assessment):
    """実際の surfaces と評価条件から UA, eta_AC, eta_AH と部位別熱損失・日射取得量を返す。"""
    region = assessment["region"]
    if type(region) is not int or not 1 <= region <= 8:
        raise ValueError("region must be an integer from 1 to 8")
    surfaces = {s["key"]: s for s in input_data["surfaces"]}
    if len(surfaces) != len(input_data["surfaces"]):
        raise ValueError("duplicate surface key")
    seen = set(assessment["excluded"])
    details = []
    window_values = {}
    for spec in assessment["parts"]:
        keys = spec["keys"]
        if not keys or len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError(f"empty or repeated assessment surface: {keys}")
        seen.update(keys)
        weights = [_number(surfaces[k]["area"], k, positive=True) for k in keys]
        total = sum(weights)
        area = total.quantize(D(".01"), rounding=ROUND_HALF_UP)
        kind = spec["kind"]
        if kind == "slab":
            details.append((keys[0], area, D(0), D(0), D(0)))
            continue
        if kind not in {"window", "door", "wall", "floor", "ceiling", "roof"}:
            raise ValueError(f"unknown kind {kind!r}")
        if kind == "door" and "U" not in spec:
            raise ValueError(f"{keys[0]}: specify whole-opening U")
        u = sum(a * _u(surfaces[k], spec) for a, k in zip(weights, keys)) / total
        q = area * u * _number(spec["H"], "H", maximum=1)
        if kind == "window":
            seasons = ("C",) if region == 8 else ("C", "H")
            # A grouped part can contain windows with different surface specifications.
            values = [window_eta(spec, surfaces[k], seasons) for k in keys]
            spec = dict(spec)
            adopted = {}
            for season in seasons:
                eta = (
                    sum(w * _number(v[season], f"{k}: eta_d", maximum=1) for k, w, v in zip(keys, weights, values))
                    / total
                )
                spec["eta_d_" + season] = eta
                adopted["eta_d_" + season] = float(eta)
            window_values[keys[0]] = adopted
        mc = area * _solar(spec, "C", region, u)
        mh = area * _solar(spec, "H", region, u) if region != 8 else D(0)
        details.append((keys[0], area, q, mc, mh))
    if seen != set(surfaces):
        raise ValueError(f"assessment/input surface mismatch: {sorted(seen.symmetric_difference(surfaces))}")
    if any(part["kind"] == "slab" for part in assessment["parts"]) and not any(
        line["kind"] == "perimeter" for line in assessment["lines"]
    ):
        raise ValueError("specify slab perimeter length and psi")
    used_wall_keys = set()
    line_details = []
    for line in assessment["lines"]:
        length, adopted_length = _line_length(line, surfaces, used_wall_keys)
        psi = _number(line["psi"], "psi")
        q = length * psi * _number(line["H"], "H", maximum=1)
        if line["kind"] not in {"perimeter", "bridge"}:
            raise ValueError("line kind must be perimeter or bridge")
        bridge = line["kind"] == "bridge"
        mc = length * _solar(line, "C", region, psi) if bridge else D(0)
        mh = length * _solar(line, "H", region, psi) if bridge and region != 8 else D(0)
        line_details.append((line["name"], D(0), q, mc, mh, adopted_length))
    if not details or not (area := sum(row[1] for row in details)):
        raise ValueError("positive envelope area required")
    rows = [(*row, window_values.get(row[0], {})) for row in details] + line_details
    q, mc, mh = (sum(row[i] for row in rows) for i in (2, 3, 4))
    return {
        "UA": float((q / area).quantize(D(".01"), rounding=ROUND_CEILING)),
        "eta_AC": float((100 * mc / area).quantize(D(".1"), rounding=ROUND_CEILING)),
        "eta_AH": float((100 * mh / area).quantize(D(".1"), rounding=ROUND_FLOOR)) if region != 8 else None,
        "A_env": float(area),
        "q": float(q),
        "m_C": float(mc),
        "m_H": float(mh) if region != 8 else None,
        "parts": [
            {
                "name": name,
                **extra,
                "area": float(a),
                "q": float(loss),
                "m_C": float(c),
                "m_H": float(h) if region != 8 else None,
            }
            for name, a, loss, c, h, extra in rows
        ],
    }

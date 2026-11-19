"""一重構造の窓。建築研究所 第三章第四節 付録C 表1・式(1a)～(1c)。"""

from decimal import Decimal
from itertools import product
from math import isfinite
import warnings

# 整数は百分率。ガラスの列順は ATTACHMENTS に対応する。
FRAMES = {"木製・樹脂製": 72, "金属製など": 80, "枠の影響なし": 100}
FRAMES.update({name: 80 for name in ("木・樹脂と金属の複合", "金属製熱遮断", "金属製")})
ATTACHMENTS = ("なし", "和障子", "内付け遮蔽", "外付け遮蔽")
GLASS = {
    "単板": (88, 38, 62, 19),
    "一般複層": (79, 38, 57, 17),
    "Low-E二層・取得型": (64, 38, 47, 15),
    "Low-E二層・遮蔽型": (40, 26, 31, 11),
    "一般三層": (72, 38, 53, 18),
    "Low-E三層・取得型": (59, 37, 44, 14),
    "Low-E三層・遮蔽型": (37, 25, 29, 10),
    "Low-E三層・複数膜・取得型": (54, 34, 41, 12),
    "Low-E三層・複数膜・遮蔽型": (33, 22, 26, 8),
}


def window_eta(spec, surface, seasons):
    """季別値、共通値、仕様表の順で決める。空欄は指定済み条件に合う候補に限定する。"""
    resolved = {}
    for season in seasons:
        value = spec.get("eta_d_" + season)
        resolved[season] = spec.get("eta_d") if value is None else value
    needed = [season for season in seasons if resolved[season] is None]
    if not needed:
        return resolved
    choices = []
    missing = []
    for field, allowed in (("frame_type", FRAMES), ("glass_type", GLASS), ("attachment", ATTACHMENTS)):
        value = spec.get(field)
        if value is None:
            value = surface.get(field)
        if value is None or value == "":
            missing.append(field)
            choices.append(list(allowed))
        elif value in allowed:
            choices.append([value])
        else:
            raise ValueError(f"{surface['key']}: {field}={value!r}; choices: {', '.join(allowed)}")
    values = [
        Decimal(FRAMES[frame] * GLASS[glass][ATTACHMENTS.index(attachment)]) / 10000
        for frame, glass, attachment in product(*choices)
    ]
    for season in needed:
        resolved[season] = max(values) if season == "C" else min(values)
    if missing:
        adopted = ", ".join(f"eta_d_{s}={resolved[s]}" for s in needed)
        warnings.warn(f"{surface['key']}: {', '.join(missing)} が未入力。候補中の不利側を使用: {adopted}", stacklevel=3)
    return resolved

# 第三章第三節 付録B 表3（一重窓）。各行は (中空層の最小厚さ mm, 窓全体の U)。
# 昇順に格納し、厚さを満たす最後の区分を採用する。
_WINDOW_U = {
    "木製・樹脂製": {
        "単板": [(0, 6.51)], "一般複層": [(6, 3.49), (10, 2.91)],
        "Low-E二層・ガス": [(4, 2.91), (8, 2.33), (12, 1.90)],
        "Low-E二層・空気": [(5, 2.91), (10, 2.33)],
        "Low-E三層・ガス": [(6, 1.70)], "Low-E三層・空気": [(9, 1.70)],
        "Low-E三層・複数膜・ガス": [(7, 1.60)],
    },
    "木・樹脂と金属の複合": {
        "一般複層": [(6, 4.07), (10, 3.49)],
        "Low-E二層・ガス": [(4, 3.49), (8, 2.33), (16, 2.15)],
        "Low-E二層・空気": [(5, 3.49), (10, 2.33)],
    },
    "金属製熱遮断": {
        "一般複層": [(6, 4.07), (10, 3.49)],
        "Low-E二層・ガス": [(4, 3.49), (8, 2.91)], "Low-E二層・空気": [(6, 3.49), (10, 2.91)],
    },
    "金属製": {
        "単板": [(0, 6.51)], "一般複層": [(4, 4.65), (10, 4.07)],
        "Low-E二層・ガス": [(4, 4.07), (8, 3.49)], "Low-E二層・空気": [(5, 4.07), (10, 3.49)],
    },
}


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: 数値が必要です") from exc
    if not isfinite(number):
        raise ValueError(f"{label}: 有限値が必要です")
    return number


def window_u(spec):
    """最終の窓全体Uを直接指定、または仕様表から求める（和障子は開閉平均）。"""
    if spec.get("U") is not None:
        u = _finite(spec["U"], "U")
        if u <= 0:
            raise ValueError("U must be positive")
        return u
    frame = spec.get("frame_type")
    glass = spec.get("glass_type")
    if frame not in _WINDOW_U or glass not in GLASS:
        raise ValueError("U算定には詳細な frame_type と glass_type、または窓全体の U を指定してください")
    family = glass
    if glass.startswith("Low-E"):
        gas = spec.get("gap_gas")
        if gas not in ("空気", "ガス"):
            raise ValueError("gap_gas に 空気 または ガス（断熱性ガス）を指定してください")
        family = glass.rsplit("・", 1)[0] + "・" + gas
    gap = 0 if glass == "単板" else _finite(spec.get("gap_mm"), "gap_mm")
    rows = _WINDOW_U[frame].get(family, [])
    for minimum, u in reversed(rows):
        if gap >= minimum:
            if spec.get("attachment") == "和障子":
                return (u + 1 / (1 / u + 0.18)) / 2
            return u
    raise ValueError(f"仕様表の適用範囲外: {frame}, {family}, {gap} mm。窓全体の U を指定してください")


def window_properties(*, frame_type, glass_type, attachment, gap_mm=None, gap_gas=None,
                      U=None, eta_d=None, convective_fraction=None, SCC=None, SCR=None):
    """同じ仕様から外皮評価用と時刻別計算用の窓入力を返す。solar は窓面入射日射。"""
    spec = {"frame_type": frame_type, "glass_type": glass_type, "attachment": attachment,
            "gap_mm": gap_mm, "gap_gas": gap_gas, "U": U}
    if frame_type not in FRAMES or glass_type not in GLASS or attachment not in ATTACHMENTS:
        raise ValueError("frame_type, glass_type, attachment を選択してください")
    u = window_u(spec)
    eta = (_finite(eta_d, "eta_d") if eta_d is not None
           else float(window_eta(spec, {"key": "窓"}, ("C",))["C"]))
    if not 0 <= eta <= 1:
        raise ValueError("eta_d must be in [0, 1]")
    if SCC is not None or SCR is not None:
        scc, scr = _finite(SCC, "SCC"), _finite(SCR, "SCR")
        if convective_fraction is not None or min(scc, scr) < 0 or scc + scr > 1 + 1e-12:
            raise ValueError("SCC・SCR は対で指定し、非負かつ合計1以下としてください")
    else:
        if convective_fraction is None:
            warnings.warn("対流割合が未入力。日射取得率だけでは SCC と SCR を決定できません", stacklevel=2)
            raise ValueError("convective_fraction または既存の SCC・SCR を指定してください")
        fraction = _finite(convective_fraction, "convective_fraction")
        if not 0 <= fraction <= 1:
            raise ValueError("convective_fraction must be in [0, 1]")
        scc = eta * fraction
        scr = eta - scc
    return {"rating": {"U": u, "eta_d": eta},
            "surface": {"whole_window_u": u, "SCC": scc, "SCR": scr}}

"""一重構造の窓。建築研究所 第三章第四節 付録C 表1・式(1a)～(1c)。"""

from decimal import Decimal
from itertools import product
import warnings

# 整数は百分率。ガラスの列順は ATTACHMENTS に対応する。
FRAMES = {"木製・樹脂製": 72, "金属製など": 80, "枠の影響なし": 100}
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

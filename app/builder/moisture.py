from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .logger import get_logger
from .utils import convert_to_json_compatible
from .validate import ConfigFileError

logger = get_logger(__name__)


def _normalize_series(value: Any, *, field: str) -> Any:
    """
    humidity_generation は number または array<number> を受ける。
    - numpy/pandas 等は list へ正規化
    - list/tuple/ndarray は list（中身はfloat化）
    - scalar は scalar（float）
    """
    v = convert_to_json_compatible(value)
    if isinstance(v, (list, tuple, np.ndarray)):
        out: List[float] = []
        for x in list(v):
            try:
                out.append(float(x))
            except Exception:
                raise ConfigFileError(f"humidity_source.{field} must be number or array<number>, got element {x!r}")
        return out
    try:
        return float(v)
    except Exception:
        raise ConfigFileError(f"humidity_source.{field} must be number or array<number>, got {value!r}")


def _series_summary(v: Any) -> str:
    if isinstance(v, list):
        if not v:
            return "series[len=0]"
        return f"series[len={len(v)} first={v[0]} last={v[-1]}]"
    return f"scalar[{v}]"


def build_humidity_generation_vents(
    *,
    raw_config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    raw_config.humidity_source / humidity_sources を ventilation_branches の「発湿（湿気生成）」へ変換して返す。

    対応入力:
    1) 新形式（推奨・heat_sourceと同系）
      "humidity_source": [
        {"key": "LD_hum", "room": "LD", "generation_rate": [..kg/s..]}
      ]

    2) 旧vtsim互換（dict）
      "humidity_source": {
        "LD_発湿": {"set": "LD", "mx": [..kg/s..]}
      }

    出力:
      ventilation_branches に
        - key: "void-><room>"
        - vol: 0.0 （空気移動は発生させない）
        - humidity_generation: series/scalar
        - subtype: "internal_humidity"
      を追加する。

    戻り値:
      (add_ventilation_branches, rooms_to_enable_calc_x)
    """
    hs = raw_config.get("humidity_source", raw_config.get("humidity_sources"))
    if hs is None:
        return [], []

    branches: List[Dict[str, Any]] = []
    rooms: List[str] = []

    # 旧vtsim互換（dict形式）
    if isinstance(hs, dict):
        for key, item in hs.items():
            if not isinstance(item, dict):
                raise ConfigFileError(f"humidity_source[{key!r}] must be dict, got {item!r}")
            room = item.get("set")
            if not isinstance(room, str) or not room:
                raise ConfigFileError(f"humidity_source[{key!r}].set must be non-empty string")
            if "mx" not in item:
                raise ConfigFileError(f"humidity_source[{key!r}] missing required field: mx")
            gen = _normalize_series(item.get("mx"), field="mx")
            branch_key = f"void->{room}"
            logger.info("　発湿(互換)換気ブランチ【%s】を追加します: key=%s rate=%s", branch_key, key, _series_summary(gen))
            branches.append(
                {
                    "key": branch_key,
                    "vol": 0.0,
                    "humidity_generation": gen,
                    "subtype": "internal_humidity",
                    "comment": str(key),
                }
            )
            rooms.append(room)
        return branches, rooms

    # 新形式（list[dict]）
    if not isinstance(hs, list):
        raise ConfigFileError("humidity_source must be list[dict] or dict")

    for i, item in enumerate(hs):
        if not isinstance(item, dict):
            raise ConfigFileError(f"humidity_source[{i}] must be dict, got {item!r}")
        key = item.get("key", f"humidity_source_{i}")
        room = item.get("room")
        if not isinstance(room, str) or not room:
            raise ConfigFileError(f"humidity_source[{i}].room must be non-empty string")
        if "generation_rate" not in item:
            raise ConfigFileError(f"humidity_source[{i}] missing required field: generation_rate")

        gen = _normalize_series(item.get("generation_rate"), field="generation_rate")
        branch_key = f"void->{room}"
        logger.info("　発湿換気ブランチ【%s】を追加します: key=%s rate=%s", branch_key, key, _series_summary(gen))
        branches.append(
            {
                "key": branch_key,
                "vol": 0.0,
                "humidity_generation": gen,
                "subtype": "internal_humidity",
                "comment": str(key),
            }
        )
        rooms.append(room)

    return branches, rooms



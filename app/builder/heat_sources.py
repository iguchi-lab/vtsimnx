from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .logger import get_logger
from .utils import convert_to_json_compatible
from .surfaces import DEFAULT_ETA_LW, get_node_prefix
from .validate import ConfigFileError

logger = get_logger(__name__)


def _as_float(value: Any, *, field: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        raise ConfigFileError(f"heat_source.{field} must be number, got {value!r}")


def _normalize_fractions(item: Dict[str, Any]) -> Tuple[float, float]:
    """
    戻り値: (convection_frac, radiation_frac)
    - 両方未指定: (1.0, 0.0)
    - 片方だけ: 残りは 1-指定値
    - 両方指定: そのまま（合計が0..1に収まらない場合はエラー）
    """
    conv = _as_float(item.get("convection"), field="convection", default=None)
    rad = _as_float(item.get("radiation"), field="radiation", default=None)

    if conv is None and rad is None:
        conv, rad = 1.0, 0.0
    elif conv is None and rad is not None:
        conv = 1.0 - rad
    elif conv is not None and rad is None:
        rad = 1.0 - conv

    assert conv is not None and rad is not None
    if conv < 0.0 or rad < 0.0 or (conv + rad) > 1.0 + 1e-9:
        raise ConfigFileError(
            f"heat_source convection+radiation must be within [0,1] and sum<=1, got convection={conv}, radiation={rad}"
        )
    return float(conv), float(rad)


def _normalize_heat_series(value: Any) -> Any:
    """
    solver は heat_generation を number または array<number> として受ける。
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
                raise ConfigFileError(
                    f"heat_source.generation_rate must be number or array<number>, got element {x!r}"
                )
        return out
    try:
        return float(v)
    except Exception:
        raise ConfigFileError(f"heat_source.generation_rate must be number or array<number>, got {value!r}")


def _series_summary(v: Any) -> str:
    """
    ログ出力用の短いサマリ。
    - 具体的な時系列値を全文出すとログが巨大になるため、型/長さ/先頭/末尾のみ出す。
    """
    if isinstance(v, list):
        if not v:
            return "series[len=0]"
        first = v[0]
        last = v[-1]
        return f"series[len={len(v)} first={first} last={last}]"
    return f"scalar[{v}]"


def _room_surfaces(surface_config: List[Dict[str, Any]], room: str) -> List[Tuple[str, float, float]]:
    """
    対象 room の「室内側表面ノード」一覧を返す。
    戻り値: [(surface_node_key, area, eta_lw), ...]
    """
    out: List[Tuple[str, float, float]] = []
    for s in surface_config or []:
        if not isinstance(s, dict):
            continue
        try:
            start_node, _end_node, i_prefix, _o_prefix = get_node_prefix(s)
        except Exception:
            continue
        if str(start_node) != str(room):
            continue
        area = s.get("area")
        try:
            a = float(area)
        except Exception:
            continue
        if a <= 0.0:
            continue
        # 長波放射の吸収率は epsilon を優先（互換で eta も許容）
        eta_lw = _as_float(s.get("epsilon", s.get("eta")), field="epsilon", default=DEFAULT_ETA_LW)
        assert eta_lw is not None
        out.append((f"{i_prefix}_s", a, float(eta_lw)))
    return out


def build_heat_generation_branches(
    *,
    raw_config: Dict[str, Any],
    surface_config: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    raw_config.heat_source を thermal_branches（heat_generation）へ変換して返す。

    入力（例）:
      "heat_source": [
        {"key": "LD_人体", "room": "LD", "generation_rate": [..W..], "convection": 0.5, "radiation": 0.5}
      ]
    """
    hs = raw_config.get("heat_source", raw_config.get("heat_sources"))
    if hs is None:
        return []
    if not isinstance(hs, list):
        raise ConfigFileError("heat_source must be list[dict]")

    branches: List[Dict[str, Any]] = []
    for i, item in enumerate(hs):
        if not isinstance(item, dict):
            raise ConfigFileError(f"heat_source[{i}] must be dict, got {item!r}")
        key = item.get("key", f"heat_source_{i}")
        room = item.get("room")
        if not isinstance(room, str) or not room:
            raise ConfigFileError(f"heat_source[{i}].room must be non-empty string")
        if "generation_rate" not in item:
            raise ConfigFileError(f"heat_source[{i}] missing required field: generation_rate")

        conv_frac, rad_frac = _normalize_fractions(item)
        q_total = _normalize_heat_series(item.get("generation_rate"))

        # 係数の適用（配列/スカラー両対応）
        def _scale(v: Any, k: float) -> Any:
            if isinstance(v, list):
                return [float(x) * k for x in v]
            return float(v) * k

        q_conv = _scale(q_total, conv_frac)
        q_rad = _scale(q_total, rad_frac)

        # 対流分: 室空間（ノード）へ投入
        if (isinstance(q_conv, list) and any(x != 0.0 for x in q_conv)) or (not isinstance(q_conv, list) and q_conv != 0.0):
            branch_key = f"void->{room}"
            logger.info(
                "　発熱(対流)熱ブランチ【%s】を追加します: src=void tgt=%s key=%s rate=%s",
                branch_key,
                room,
                key,
                _series_summary(q_conv),
            )
            branches.append(
                {
                    "key": branch_key,
                    "heat_generation": q_conv,
                    "subtype": "internal_convection",
                    "comment": str(key),
                }
            )

        # 放射分: 対象室の表面へ面積按分（surfaces が無ければ室へ）
        surfaces = _room_surfaces(surface_config, room)
        if surfaces:
            sum_area = sum(a for _k, a, _eta in surfaces)
            if sum_area <= 0.0:
                surfaces = []
        if (isinstance(q_rad, list) and any(x != 0.0 for x in q_rad)) or (not isinstance(q_rad, list) and q_rad != 0.0):
            if not surfaces:
                branch_key = f"void->{room}"
                logger.info(
                    "　発熱(放射)熱ブランチ【%s】を追加します: src=void tgt=%s key=%s rate=%s (surfacesなし→室へ集約)",
                    branch_key,
                    room,
                    key,
                    _series_summary(q_rad),
                )
                branches.append(
                    {
                        "key": branch_key,
                        "heat_generation": q_rad,
                        "subtype": "internal_radiation",
                        "comment": str(key),
                    }
                )
            else:
                for surf_node, area, eta_lw in surfaces:
                    frac = area / sum_area
                    branch_key = f"void->{surf_node}"
                    q_abs = _scale(q_rad, frac * eta_lw)
                    logger.info(
                        "　発熱(放射)熱ブランチ【%s】を追加します: src=void tgt=%s room=%s key=%s rate=%s area=%.6g/%.6g frac=%.6g eta=%.6g",
                        branch_key,
                        surf_node,
                        room,
                        key,
                        _series_summary(q_abs),
                        area,
                        sum_area,
                        frac,
                        eta_lw,
                    )
                    branches.append(
                        {
                            "key": branch_key,
                            "heat_generation": q_abs,
                            "subtype": "internal_radiation",
                            "comment": f"{key} (area_fraction={frac:.6g}, eta={eta_lw:.6g})",
                        }
                    )

        logger.info("発熱を追加しました: key=%s room=%s (conv=%.3f, rad=%.3f)", key, room, conv_frac, rad_frac)

    return branches



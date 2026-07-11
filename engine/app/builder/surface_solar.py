from __future__ import annotations

import numpy as np

from .logger import get_logger
from .surface_constants import DEFAULT_EPSILON_LW, DEFAULT_ETA_SW
from .surface_layers import _branch_log_detail, get_node_prefix
from .surface_radiation import collect_room_side_surfaces
from .utils import CHAIN_DELIMITER, ensure_timeseries

logger = get_logger(__name__)


def process_wall_solar(surface: dict, sim_length: int, verbose: bool = True) -> list:
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    heat_generation = surface["area"] * surface.get("eta", DEFAULT_ETA_SW) * np.array(surface["solar"])
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
    if verbose:
        logger.info(f"　外壁日射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
    thermal_branches.append(b)
    return thermal_branches


def process_wall_nocturnal(surface: dict, sim_length: int, verbose: bool = True) -> list:
    """
    夜間放射（長波放射）による熱損失を、void から表面ノードへの heat_generation として表現する。

    注意:
    - 熱ブランチは target が必ず実在ノードである必要があるため、`surface->void` は作れない。
      代わりに `void->surface` を作り、heat_generation を負にして「表面から void へ流出」を表す。
    - nocturnal は [W/m2]（または入力系列と同単位）を想定し、面積を掛けて [W] の系列にする。
    """
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    # 設定キーは "nocturnal" を推奨。互換で "night_radiation" も受ける。
    noct = surface.get("nocturnal", surface.get("night_radiation"))
    if noct is None:
        return thermal_branches

    # 表面->void への流出なので負符号。夜間放射は長波なので epsilon=0.9
    heat_generation = -surface["area"] * surface.get("epsilon", DEFAULT_EPSILON_LW) * np.array(noct)
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "nocturnal_loss"}
    if verbose:
        logger.info(f"　外壁夜間放射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
    thermal_branches.append(b)
    return thermal_branches


def process_glass_solar(surface: dict, surfaces: list, sim_length: int, verbose: bool = True) -> list:
    thermal_branches: list = []

    # NOTE:
    # - 同一室（start_node が同じ）の表面に対して配分する。
    # - startswith だと "Room1" と "Room10" のような前方一致で誤って混ざるため、
    #   CHAIN_DELIMITER で分割した先頭ノードの「完全一致」で判定する。
    node = str(surface["key"]).split(CHAIN_DELIMITER, 1)[0]
    # 基準室 node に接する表面（start/end 両側）を収集する。
    # これにより "X->LD" のような面でも LD 側表面ノードを配分対象に含められる。
    room_side_surfaces = collect_room_side_surfaces(node, surfaces)

    area_ceiling = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "ceiling"])
    area_wall = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "wall"])
    area_ceiling_wall = area_ceiling + area_wall
    area_floor = sum([area for _s, _node_key, part, area in room_side_surfaces if part == "floor"])

    # ガラス透過日射の配分:
    # - 床/床以外（壁・天井）: eta の代わりに SCR を掛けて表面ノードへ投入
    # - 室空間（空気ノード）   : SCC を掛けて投入（追加ブランチ）
    #
    # 互換: 既存入力が eta のみの場合は SCR のデフォルトとして eta を使用する。日射は短波なので 0.8。
    scr = surface.get("SCR", surface.get("scr", surface.get("eta", DEFAULT_ETA_SW)))
    scc = surface.get("SCC", surface.get("scc", 0.0))

    base = np.array(surface["solar"]) * surface["area"]
    heat_generation_floor        = base * 0.50 * scr
    heat_generation_ceiling_wall = base * 0.50 * scr
    heat_generation_space        = base * scc

    heat_generation_floor        = ensure_timeseries(heat_generation_floor,        sim_length)
    heat_generation_ceiling_wall = ensure_timeseries(heat_generation_ceiling_wall, sim_length)
    heat_generation_space        = ensure_timeseries(heat_generation_space,        sim_length)

    for s, room_node_key, part, area in room_side_surfaces:
        branch_key = f"void->{room_node_key}"
        # 室内側の各面での「日射吸収」を表すため、受け側表面の eta を掛ける（短波なので 0.8）
        eta_abs = float(s.get("eta", DEFAULT_ETA_SW))
        if part == "floor":
            if area_floor <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_floor) * eta_abs * area / area_floor
            ).tolist()
        elif part == "ceiling" or part == "wall":
            if area_ceiling_wall <= 0:
                continue
            heat_generation = (
                np.array(heat_generation_ceiling_wall) * eta_abs * area / area_ceiling_wall
            ).tolist()
        else:
            continue
        b = {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
        if verbose:
            logger.info(f"　ガラス透過日射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
        thermal_branches.append(b)

    # 室空間（ノード）へ SCC 分を追加投入
    # key は一旦 "void->{node}" として生成し、重複があれば validation 側で (01),(02)... にリネームされる。
    if any(v != 0.0 for v in heat_generation_space):
        branch_key = f"void->{node}"
        b = {
            "key": branch_key,
            "heat_generation": list(heat_generation_space),
            "subtype": "solar_gain",
            "comment": "glass_solar_space(SCC)",
        }
        if verbose:
            logger.info(f"　ガラス透過日射（室空間SCC）熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
        thermal_branches.append(b)

    return thermal_branches

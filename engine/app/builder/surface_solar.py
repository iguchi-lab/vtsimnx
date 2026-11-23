from __future__ import annotations

import numpy as np

from .logger import get_logger
from .surface_constants import DEFAULT_EPSILON_LW, DEFAULT_ETA_SW
from .surface_layers import _branch_log_detail, get_node_prefix
from .surface_radiation import collect_room_side_surfaces
from .utils import CHAIN_DELIMITER, ensure_timeseries

logger = get_logger(__name__)

# 窓透過日射の SCR 室内配分は保存的配分のみ（AE-Sim/Heat §6.8.4 相当）。
# scr_distribution は互換・明示用。省略時も同じ。非保存的な旧経路は不可。
SCR_DISTRIBUTION_SIMHEAT = "simheat"
_VALID_SCR_DISTRIBUTIONS = {SCR_DISTRIBUTION_SIMHEAT}


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


def _scr_distribution_mode(surface: dict) -> str:
    """常に保存的配分。明示値は simheat のみ可（省略も同じ）。"""
    raw = surface.get("scr_distribution")
    if raw is None:
        return SCR_DISTRIBUTION_SIMHEAT
    mode = str(raw).strip().lower()
    if mode not in _VALID_SCR_DISTRIBUTIONS:
        raise ValueError(
            f"{surface.get('key', '?')}: scr_distribution は "
            f"{sorted(_VALID_SCR_DISTRIBUTIONS)} または省略のみです（got {raw!r}）。"
            "放射成分を受熱面 eta でさらに減衰させる旧配分は不可です。"
        )
    return mode


def _group_weight_sum(faces: list[tuple[dict, str, str, float]], *, use_absorptance: bool) -> float:
    total = 0.0
    for s, _node, _part, area in faces:
        if use_absorptance:
            total += area * float(s.get("eta", DEFAULT_ETA_SW))
        else:
            total += area
    return total


def _allocate_group_heat(
    group_heat: np.ndarray,
    faces: list[tuple[dict, str, str, float]],
) -> list[tuple[str, list[float]]]:
    """1群分の放射熱を A*η 重み（ゼロなら面積比）で配分し、(node_key, series) を返す。"""
    if not faces or not np.any(group_heat):
        return []

    weight_sum = _group_weight_sum(faces, use_absorptance=True)
    use_absorptance = True
    if weight_sum <= 0.0:
        weight_sum = _group_weight_sum(faces, use_absorptance=False)
        use_absorptance = False
        if weight_sum <= 0.0:
            return []

    out: list[tuple[str, list[float]]] = []
    for s, room_node_key, _part, area in faces:
        eta_abs = float(s.get("eta", DEFAULT_ETA_SW))
        weight = area * eta_abs if use_absorptance else area
        out.append((room_node_key, (group_heat * weight / weight_sum).tolist()))
    return out


def process_glass_solar(surface: dict, surfaces: list, sim_length: int, verbose: bool = True) -> list:
    thermal_branches: list = []

    # NOTE:
    # - 同一室（start_node が同じ）の表面に対して配分する。
    # - startswith だと "Room1" と "Room10" のような前方一致で誤って混ざるため、
    #   CHAIN_DELIMITER で分割した先頭ノードの「完全一致」で判定する。
    _ = _scr_distribution_mode(surface)  # 不正指定を拒否
    node = str(surface["key"]).split(CHAIN_DELIMITER, 1)[0]
    # 基準室 node に接する表面（start/end 両側）を収集する。
    room_side_surfaces = collect_room_side_surfaces(node, surfaces)

    floor_faces = [item for item in room_side_surfaces if item[2] == "floor"]
    other_faces = [item for item in room_side_surfaces if item[2] in ("ceiling", "wall")]

    # ガラス透過日射:
    # Q_rad = Q_tr * SCR を床50%/その他50%（片群なら全量）へ、群内は A*η で保存的配分。
    # Q_conv = Q_tr * SCC は室空気へ。
    # 互換: 既存入力が eta のみの場合は SCR のデフォルトとして eta を使用する。
    scr = surface.get("SCR", surface.get("scr", surface.get("eta", DEFAULT_ETA_SW)))
    scc = surface.get("SCC", surface.get("scc", 0.0))

    base = np.asarray(surface["solar"], dtype=float) * float(surface["area"])
    q_rad = base * float(scr)
    q_conv = base * float(scc)

    if np.any(q_rad):
        if not floor_faces and not other_faces:
            raise ValueError(
                f"{surface['key']}: SCR を受ける床・壁・天井の吸収面が必要です"
                f"（室 {node}）"
            )

        if not floor_faces:
            floor_share = 0.0
        elif not other_faces:
            floor_share = 1.0
        else:
            floor_share = 0.50

        heat_floor = q_rad * floor_share
        heat_other = q_rad * (1.0 - floor_share)

        for room_node_key, series in _allocate_group_heat(heat_floor, floor_faces):
            b = {
                "key": f"void->{room_node_key}",
                "heat_generation": ensure_timeseries(series, sim_length),
                "subtype": "solar_gain",
            }
            if verbose:
                logger.info(
                    f"　ガラス透過日射熱ブランチ【{b['key']}】を追加します。{_branch_log_detail(b)}"
                )
            thermal_branches.append(b)

        for room_node_key, series in _allocate_group_heat(heat_other, other_faces):
            b = {
                "key": f"void->{room_node_key}",
                "heat_generation": ensure_timeseries(series, sim_length),
                "subtype": "solar_gain",
            }
            if verbose:
                logger.info(
                    f"　ガラス透過日射熱ブランチ【{b['key']}】を追加します。{_branch_log_detail(b)}"
                )
            thermal_branches.append(b)

    heat_generation_space = ensure_timeseries(q_conv, sim_length)
    if any(v != 0.0 for v in heat_generation_space):
        branch_key = f"void->{node}"
        b = {
            "key": branch_key,
            "heat_generation": list(heat_generation_space),
            "subtype": "solar_gain",
            "comment": "glass_solar_space(SCC)",
        }
        if verbose:
            logger.info(
                f"　ガラス透過日射（室空間SCC）熱ブランチ【{branch_key}】を追加します。"
                f"{_branch_log_detail(b)}"
            )
        thermal_branches.append(b)

    return thermal_branches

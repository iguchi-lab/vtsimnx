from __future__ import annotations

from .surface_constants import DEFAULT_ALPHA_R, SURFACE_PAIR
from .surface_layers import _branch_log_detail, _get_surface_part, get_node_prefix
from .logger import get_logger

logger = get_logger(__name__)


def collect_room_side_surfaces(
    room: str,
    surfaces: list[dict],
    *,
    exclude_glass: bool = False,
) -> list[tuple[dict, str, str, float]]:
    """
    基準室 `room` に接する表面（start/end 両側）を収集する。
    戻り値: [(surface_dict, room_side_node_key, room_side_part, area), ...]
    """
    out: list[tuple[dict, str, str, float]] = []
    room_s = str(room)
    for s in surfaces or []:
        if not isinstance(s, dict):
            continue
        try:
            part_start = _get_surface_part(s)
        except Exception:
            continue
        if exclude_glass and part_start == "glass":
            continue
        try:
            start_node, end_node, i_prefix, o_prefix = get_node_prefix(s)
        except Exception:
            continue
        try:
            area = float(s.get("area", 0.0))
        except Exception:
            continue
        if area <= 0.0:
            continue

        part_end = SURFACE_PAIR.get(part_start)

        if str(start_node) == room_s:
            out.append((s, f"{i_prefix}_s", part_start, area))
        # A->A の自己ループ面は二重計上しない
        if str(end_node) == room_s and str(end_node) != str(start_node) and part_end is not None:
            out.append((s, f"{o_prefix}_s", part_end, area))
    return out


def _collect_room_to_node_area(
    surface_data: list[dict],
    *,
    exclude_glass: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    """
    表面リストを 1 パスで走査し、室ごとに (node_key, area) のリストを集計する。
    戻り値: {room: [(node_key, area), ...], ...}（同一 node_key は未集約。process_radiation 用に集約する側で sum 可能）。
    室内放射の room ループで collect_room_side_surfaces を部屋数回呼ぶ代わりに 1 パスで済ませる。
    """
    room_to_list: dict[str, list[tuple[str, str, float]]] = {}  # room -> [(node_key, part, area), ...]

    for s in surface_data or []:
        if not isinstance(s, dict):
            continue
        try:
            part_start = _get_surface_part(s)
        except Exception:
            continue
        if exclude_glass and part_start == "glass":
            continue
        try:
            start_node, end_node, i_prefix, o_prefix = get_node_prefix(s)
        except Exception:
            continue
        try:
            area = float(s.get("area", 0.0))
        except Exception:
            continue
        if area <= 0.0:
            continue

        part_end = SURFACE_PAIR.get(part_start)
        start_s = str(start_node)
        end_s = str(end_node)

        if start_s not in room_to_list:
            room_to_list[start_s] = []
        room_to_list[start_s].append((f"{i_prefix}_s", part_start, area))
        if end_s != start_s and part_end is not None:
            if end_s not in room_to_list:
                room_to_list[end_s] = []
            room_to_list[end_s].append((f"{o_prefix}_s", part_end, area))

    # 室ごとに node_key で面積を集約して (node_key, area) のリストに
    result: dict[str, list[tuple[str, float]]] = {}
    for room, items in room_to_list.items():
        agg: dict[str, float] = {}
        for node_key, _part, area in items:
            agg[node_key] = agg.get(node_key, 0.0) + area
        result[room] = list(agg.items())
    return result


def process_radiation(node: str, surface_nodes: list[tuple[str, float]], verbose: bool = True) -> list:
    thermal_branches: list = []
    if len(surface_nodes) < 2:
        return thermal_branches
    sum_area = float(sum(a for _, a in surface_nodes))
    if sum_area <= 0.0:
        return thermal_branches

    for i, node1 in enumerate(surface_nodes):
        for j, node2 in enumerate(surface_nodes[i + 1 :], start=i + 1):
            node1_key, area1 = node1
            node2_key, area2 = node2
            branch_key = f"{node1_key}->{node2_key}"
            # 4.7 には既に両面の放射率0.9が含まれるため、室内表面間では eta を掛けない
            conductance = DEFAULT_ALPHA_R * area1 * area2 / sum_area
            b = {"key": branch_key, "conductance": conductance, "subtype": "radiation"}
            if verbose:
                logger.info(f"　室内放射熱ブランチ【{branch_key}】を追加します。{_branch_log_detail(b)}")
            thermal_branches.append(b)

    return thermal_branches

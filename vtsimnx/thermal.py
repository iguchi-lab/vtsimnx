from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)


def add_capacity(node: dict, time_step: float) -> tuple[list, list]:
    """熱容量を追加する"""
    nodes: list = []
    thermal_branches: list = []

    logger.info(f"　熱容量ノード【{node['key']}_c】を追加します。")
    nodes.append(
        {
            "key": f"{node['key']}_c",
            "calc_t": False,
            "type": "capacity",
            "ref_node": node["key"],
        }
    )

    logger.info(f"　熱容量ブランチ【{node['key']}_c->{node['key']}】を追加します。")
    thermal_branches.append(
        {
            "key": f"{node['key']}_c->{node['key']}",
            "conductance": node["thermal_mass"] / time_step,
            "subtype": "capacity",
        }
    )
    return nodes, thermal_branches



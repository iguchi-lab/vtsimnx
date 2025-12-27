from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)


def _scalar_initial_temperature(value):
    """
    ノード設定の `t`（スカラー or 時系列）から「初期値（スカラー）」を取り出す。
    熱容量ノードは通常 calc_t=False だが、初期状態の整合のため親ノードの初期値を引き継ぐ。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return None


def add_capacity(node: dict, time_step: float) -> tuple[list, list]:
    """熱容量を追加する"""
    nodes: list = []
    thermal_branches: list = []

    logger.info(f"　熱容量ノード【{node['key']}_c】を追加します。")
    init_t = _scalar_initial_temperature(node.get("t"))
    nodes.append(
        {
            "key": f"{node['key']}_c",
            "calc_t": False,
            "type": "capacity",
            "ref_node": node["key"],
            **({"t": init_t} if init_t is not None else {}),
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


def process_capacities(node_config: list, time_step: float) -> tuple[list, list]:
    """
    ノード配列を走査し、thermal_mass を持つノードに熱容量ノード/ブランチを付与。
    付与後は元のノードの thermal_mass フィールドを削除する。
    戻り値は (add_nodes, add_thermal_branches)。
    """
    add_nodes_all: list = []
    add_tb_all: list = []
    logger.info("熱容量を追加します")
    for node in node_config:
        if "thermal_mass" in node:
            add_nodes, add_tb = add_capacity(node, time_step)
            add_nodes_all.extend(add_nodes)
            add_tb_all.extend(add_tb)
            node.pop("thermal_mass", None)
    logger.info("熱容量の追加が完了しました。")
    return add_nodes_all, add_tb_all



from __future__ import annotations

from .logger import get_logger
from .surface_layers import _scalar_initial_temperature

logger = get_logger(__name__)

# archenv と揃えた乾き空気の ρ·cp [J/(m³·K)]
_DRY_AIR_RHO_CP = 1.2 * 1006.0


def add_capacity(node: dict, time_step: float) -> tuple[list, list]:
    """熱容量を追加する。

    v>0 のとき空気質量 ρ·cp·V を subtype=air_capacity として分離し、
    残り（家具等）を従来の subtype=capacity とする。
    """
    nodes: list = []
    thermal_branches: list = []

    thermal_mass = float(node["thermal_mass"])
    volume = float(node.get("v") or 0.0)
    air_mass = _DRY_AIR_RHO_CP * volume if volume > 0.0 else 0.0
    if air_mass > thermal_mass:
        air_mass = thermal_mass
    furniture_mass = max(0.0, thermal_mass - air_mass)
    init_t = _scalar_initial_temperature(node.get("t"))

    if furniture_mass > 0.0:
        logger.info(f"　熱容量ノード【{node['key']}_c】を追加します。")
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
                "conductance": furniture_mass / time_step,
                "subtype": "capacity",
            }
        )

    if air_mass > 0.0:
        logger.info(f"　空気熱容量ノード【{node['key']}_air】を追加します。")
        nodes.append(
            {
                "key": f"{node['key']}_air",
                "calc_t": False,
                "type": "capacity",
                "ref_node": node["key"],
                **({"t": init_t} if init_t is not None else {}),
            }
        )
        logger.info(f"　空気熱容量ブランチ【{node['key']}_air->{node['key']}】を追加します。")
        thermal_branches.append(
            {
                "key": f"{node['key']}_air->{node['key']}",
                "conductance": air_mass / time_step,
                "subtype": "air_capacity",
            }
        )

    # v==0 など空気質量が無い場合は従来どおり1本
    if furniture_mass <= 0.0 and air_mass <= 0.0 and thermal_mass > 0.0:
        logger.info(f"　熱容量ノード【{node['key']}_c】を追加します。")
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
                "conductance": thermal_mass / time_step,
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

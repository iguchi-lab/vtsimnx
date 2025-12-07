from __future__ import annotations

import numpy as np

from .logger import get_logger
from .utils import CHAIN_DELIMITER, ensure_timeseries

logger = get_logger(__name__)

# ------------------------------
# 表面の種類の対応関係 と 物性定数
# ------------------------------
SURFACE_PAIR = {
    "wall": "wall",
    "floor": "ceiling",
    "ceiling": "floor",
    "glass": "glass",
}

DEFAULT_ALPHA_I = 4.4   # 室内側表面の対流熱伝達率
DEFAULT_ALPHA_O = 17.9  # 室外側表面の対流熱伝達率
DEFAULT_ALPHA_R = 4.6   # 放射熱伝達率


def get_node_prefix(surface: dict) -> tuple[str, str, str, str]:
    start_node     = surface["key"].split(CHAIN_DELIMITER)[0]
    end_node       = surface["key"].split(CHAIN_DELIMITER)[1]
    start_part     = surface["part"]
    end_part       = SURFACE_PAIR[start_part]
    comment        = surface.get("comment", "").strip()
    comment_suffix = f"({comment})" if comment else ""
    i_prefix       = f"{start_node}-{end_node}{comment_suffix}_{start_part}"
    o_prefix       = f"{end_node}-{start_node}{comment_suffix}_{end_part}"
    return start_node, end_node, i_prefix, o_prefix


def process_surface(surface: dict) -> tuple[list, list]:
    nodes: list = []
    thermal_branches: list = []

    start_node, end_node, i_prefix, o_prefix = get_node_prefix(surface)

    a = surface["area"]
    alpha_i = surface.get("alpha_i", DEFAULT_ALPHA_I)
    alpha_o = surface.get("alpha_o", DEFAULT_ALPHA_O)

    if "layers" in surface:
        n = len(surface["layers"])
        q = [a * layer["lambda"] / layer["t"] for layer in surface["layers"]]
        c = [a * layer["v_capa"] * layer["t"] for layer in surface["layers"]]
    else:
        n = 1
        q = [a * surface["u_value"]]
        c = [a * surface.get("a_capacity", 0.0)]

    thermal_mass = (
        [c[0] / 2] + [c[i] / 2 + c[i + 1] / 2 for i in range(n - 1)] + [c[-1] / 2]
    )
    node_types = ["surface"] + ["internal"] * (n - 1) + ["surface"]
    conductance = [a * alpha_i] + [q[i] for i in range(n)] + [a * alpha_o]
    branch_types = ["convection"] + ["conduction"] * (n) + ["convection"]

    node_names = (
        [f"{i_prefix}_s"]
        + [f"{i_prefix}_{i+1}-{i+2}" for i in range(n - 1)]
        + [f"{o_prefix}_s"]
    )

    thermal_node_chain = [start_node] + node_names + [end_node]
    thermal_branch_names = [
        f"{thermal_node_chain[i]}->{thermal_node_chain[i+1]}" for i in range(n + 2)
    ]

    for i, node in enumerate(node_names):
        logger.info(f"　ノード【{node}】 を追加します。")
        nodes.append(
            {"key": node, "calc_t": True, "thermal_mass": thermal_mass[i], "type": "layer", "subtype": node_types[i]}
        )

    for i, branch in enumerate(thermal_branch_names):
        logger.info(f"　熱ブランチ【{branch}】を追加します。")
        thermal_branches.append(
            {"key": branch, "conductance": conductance[i], "subtype": branch_types[i]}
        )

    return nodes, thermal_branches


def process_wall_solar(surface: dict, sim_length: int) -> list:
    thermal_branches: list = []
    _, _, _, o_prefix = get_node_prefix(surface)

    heat_generation = surface["area"] * surface.get("eta", 1.0) * np.array(surface["solar"])
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    branch_key = f"void->{o_prefix}_s"
    logger.info(f"　外壁日射熱ブランチ【{branch_key}】を追加します。")
    thermal_branches.append(
        {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
    )
    return thermal_branches


def process_glass_solar(surface: dict, surfaces: list, sim_length: int) -> list:
    thermal_branches: list = []

    node = surface["key"].split(CHAIN_DELIMITER)[0]
    surfaces_of_target_node = [s for s in surfaces if s["key"].startswith(node)]
    area_ceiling = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "ceiling"])
    area_wall = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "wall"])
    area_ceiling_wall = area_ceiling + area_wall
    area_floor = sum([s["area"] for s in surfaces_of_target_node if s["part"] == "floor"])

    heat_generation_floor        = np.array(surface["solar"]) * surface["area"] * 0.50 * surface["eta"]
    heat_generation_ceiling_wall = np.array(surface["solar"]) * surface["area"] * 0.50 * surface["eta"]

    heat_generation_floor        = ensure_timeseries(heat_generation_floor,        sim_length)
    heat_generation_ceiling_wall = ensure_timeseries(heat_generation_ceiling_wall, sim_length)

    for s in surfaces_of_target_node:
        _, _, i_prefix, _ = get_node_prefix(s)
        branch_key = f"void->{i_prefix}_s"
        if s["part"] == "floor":
            heat_generation = (np.array(heat_generation_floor) * s["area"] / area_floor).tolist()
        elif s["part"] == "ceiling" or s["part"] == "wall":
            heat_generation = (np.array(heat_generation_ceiling_wall) * s["area"] / area_ceiling_wall).tolist()
        else:
            continue
        logger.info(f"　ガラス透過日射熱ブランチ【{branch_key}】を追加します。")
        thermal_branches.append(
            {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
        )

    return thermal_branches


def process_radiation(node: str, surfaces: list) -> list:
    thermal_branches: list = []
    surface_nodes = [f"{get_node_prefix(s)[2]}_s" for s in surfaces]
    sum_area = sum([s["area"] for s in surfaces])

    for i, node1 in enumerate(surface_nodes):
        for j, node2 in enumerate(surface_nodes[i + 1 :], start=i + 1):
            branch_key = f"{node1}->{node2}"
            area1 = surfaces[i]["area"]
            area2 = surfaces[j]["area"]
            conductance = DEFAULT_ALPHA_R * area1 * area2 / sum_area
            logger.info(f"　室内放射熱ブランチ【{branch_key}】を追加します。")
            thermal_branches.append(
                {"key": branch_key, "conductance": conductance, "subtype": "radiation"}
            )

    return thermal_branches


def process_surfaces(
    surface_config: list,
    sim_length: int,
    add_solar: bool = True,
    add_radiation: bool = True,
) -> tuple[list, list]:
    """
    builder から呼び出す統合処理。
    - 各面の要素分割ノード/熱ブランチを生成
    - 日射（壁/床/天井・ガラス）の熱ブランチを追加（add_solar が True の場合）
    - 室内放射の熱ブランチを追加（add_radiation が True の場合）
    戻り値は (add_nodes, add_thermal_branches)。
    """
    if not surface_config:
        return [], []

    nodes: list = []
    thermal_branches: list = []

    surface_data = surface_config

    # 表面の分解
    logger.info("表面の解析を開始します。")
    for s in surface_data:
        add_nodes, add_tb = process_surface(s)
        nodes.extend(add_nodes)
        thermal_branches.extend(add_tb)
    logger.info("表面の解析が完了しました。")

    # 日射
    if add_solar:
        logger.info("日射の解析を開始します。")
        for s in (x for x in surface_data if "solar" in x):
            if s["part"] in ["wall", "floor", "ceiling"]:
                thermal_branches.extend(process_wall_solar(s, sim_length))
            elif s["part"] == "glass":
                thermal_branches.extend(process_glass_solar(s, surface_data, sim_length))
        logger.info("日射の解析が完了しました。")
    else:
        logger.info("日射の解析をスキップします。")

    # 室内放射
    if add_radiation:
        logger.info("室内放射の解析を開始します。")
        for node in {s["key"].split(CHAIN_DELIMITER)[0] for s in surface_data}:
            surfaces = [s for s in surface_data if s["key"].startswith(node)]
            thermal_branches.extend(process_radiation(node, surfaces))
        logger.info("室内放射の解析が完了しました。")
    else:
        logger.info("室内放射の解析をスキップします。")

    return nodes, thermal_branches



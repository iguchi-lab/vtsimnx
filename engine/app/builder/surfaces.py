from __future__ import annotations

from .logger import get_logger
from .surface_constants import (
    DEFAULT_AIR_V_CAPA,
    DEFAULT_ALPHA_I,
    DEFAULT_ALPHA_O,
    DEFAULT_ALPHA_R,
    DEFAULT_EPSILON_LW,
    DEFAULT_ETA_LW,
    DEFAULT_ETA_SW,
    NOCTURNAL_TARGET_PARTS,
    SOLAR_TARGET_PARTS,
    SURFACE_PAIR,
    SURFACE_PART_ALIASES,
)
from .surface_layers import (
    _apply_hollow_layer,
    _apply_normal_layer,
    _apply_ventilated_layer,
    _branch_log_detail,
    _build_layer_node_dict,
    _get_air_v_capa,
    _get_surface_part,
    _layer_flag,
    _layer_float,
    _leading_node_key_from_layer_key,
    _node_log_detail,
    _process_surface_u_value,
    _scalar_initial_temperature,
    get_node_prefix,
    process_surface,
)
from .surface_radiation import (
    _collect_room_to_node_area,
    collect_room_side_surfaces,
    process_radiation,
)
from .surface_rc import _build_rc_continuous_abcd, _layers_to_rc_arrays
from .surface_response import (
    _auto_response_coefficients_from_layers,
    _process_surface_response_method,
)
from .surface_solar import process_glass_solar, process_wall_nocturnal, process_wall_solar

logger = get_logger(__name__)


def process_surfaces(
    surface_config: list,
    sim_length: int,
    node_config: list | None = None,
    add_solar: bool = True,
    add_nocturnal: bool = True,
    add_radiation: bool = True,
    radiation_exclude_glass: bool = False,
    layer_method: str = "rc",
    time_step: float | None = None,
    response_method: str = "arx_rc",
    response_terms: int | None = None,
    verbose: bool = True,
) -> tuple[list, list]:
    """
    builder から呼び出す統合処理。
    - 各面の要素分割ノード/熱ブランチを生成
    - 日射（壁/床/天井・ガラス）の熱ブランチを追加（add_solar が True の場合）
    - 室内放射の熱ブランチを追加（add_radiation が True の場合）
    戻り値は (add_nodes, add_thermal_branches)。
    verbose=False にするとノード/ブランチごとの詳細ログを出さず処理を軽くする。
    """
    if not surface_config:
        return [], []

    nodes: list = []
    thermal_branches: list = []

    surface_data = surface_config
    initial_t_by_node_key: dict[str, float] = {}
    if node_config:
        for n in node_config:
            if not isinstance(n, dict):
                continue
            k = n.get("key")
            if not k:
                continue
            if "t" not in n:
                continue
            init_t = _scalar_initial_temperature(n.get("t"))
            if init_t is None:
                continue
            initial_t_by_node_key[str(k)] = init_t

    # 表面の分解
    logger.info("表面の解析を開始します。")
    for s in surface_data:
        # 全体指定（引数）をデフォルトとして surface ごとに持たせる
        if isinstance(s, dict) and "layer_method" not in s:
            s["layer_method"] = layer_method
        add_nodes, add_tb = process_surface(
            s,
            initial_t_by_node_key=initial_t_by_node_key,
            time_step=time_step,
            response_method=response_method,
            response_terms=response_terms,
            verbose=verbose,
        )
        nodes.extend(add_nodes)
        thermal_branches.extend(add_tb)
    logger.info("表面の解析が完了しました。")

    # 日射
    if add_solar:
        logger.info("日射の解析を開始します。")
        for s in (x for x in surface_data if "solar" in x):
            part = _get_surface_part(s)
            if part in SOLAR_TARGET_PARTS:
                thermal_branches.extend(process_wall_solar(s, sim_length, verbose=verbose))
            elif part == "glass":
                thermal_branches.extend(process_glass_solar(s, surface_data, sim_length, verbose=verbose))
        logger.info("日射の解析が完了しました。")
    else:
        logger.info("日射の解析をスキップします。")

    # 夜間放射（外部への放射損失）
    if add_nocturnal:
        logger.info("夜間放射の解析を開始します。")
        for s in (x for x in surface_data if ("nocturnal" in x or "night_radiation" in x)):
            part = _get_surface_part(s)
            if part in NOCTURNAL_TARGET_PARTS:
                thermal_branches.extend(process_wall_nocturnal(s, sim_length, verbose=verbose))
        logger.info("夜間放射の解析が完了しました。")
    else:
        logger.info("夜間放射の解析をスキップします。")

    # 室内放射（室ごとの node_key–面積を 1 パスで集計してから放射ブランチを追加）
    if add_radiation:
        logger.info("室内放射の解析を開始します。")
        room_to_node_area = _collect_room_to_node_area(
            surface_data, exclude_glass=radiation_exclude_glass
        )
        for node, node_surfaces in room_to_node_area.items():
            if node_surfaces:
                thermal_branches.extend(process_radiation(node, node_surfaces, verbose=verbose))
        logger.info("室内放射の解析が完了しました。")
    else:
        logger.info("室内放射の解析をスキップします。")

    return nodes, thermal_branches


__all__ = [
    "DEFAULT_AIR_V_CAPA",
    "DEFAULT_ALPHA_I",
    "DEFAULT_ALPHA_O",
    "DEFAULT_ALPHA_R",
    "DEFAULT_EPSILON_LW",
    "DEFAULT_ETA_LW",
    "DEFAULT_ETA_SW",
    "NOCTURNAL_TARGET_PARTS",
    "SOLAR_TARGET_PARTS",
    "SURFACE_PAIR",
    "SURFACE_PART_ALIASES",
    "_apply_hollow_layer",
    "_apply_normal_layer",
    "_apply_ventilated_layer",
    "_auto_response_coefficients_from_layers",
    "_branch_log_detail",
    "_build_layer_node_dict",
    "_build_rc_continuous_abcd",
    "_collect_room_to_node_area",
    "_get_air_v_capa",
    "_get_surface_part",
    "_layer_flag",
    "_layer_float",
    "_layers_to_rc_arrays",
    "_leading_node_key_from_layer_key",
    "_node_log_detail",
    "_process_surface_response_method",
    "_process_surface_u_value",
    "_scalar_initial_temperature",
    "collect_room_side_surfaces",
    "get_node_prefix",
    "process_glass_solar",
    "process_radiation",
    "process_surface",
    "process_surfaces",
    "process_wall_nocturnal",
    "process_wall_solar",
]

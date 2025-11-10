from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy
import json

from .logger import get_logger
from .config_types import SimConfigType, IndexType, ToleranceType, CalcFlagType
from .utils import CHAIN_DELIMITER
from .parsers import _parse_simulation, _parse_nodes, _parse_chain_branches, _parse_surface, _parse_aircon
from .surfaces import process_surface, process_wall_solar, process_glass_solar, process_radiation
from .aircon import process_aircon
from .thermal import add_capacity
from .validate import validate_dict

logger = get_logger(__name__)


# ------------------------------
# エントリポイント
# ------------------------------
def build_config(raw_config: Dict[str, Any], output_path: Optional[str] = "parsed_input_data.json") -> Dict[str, Any]:
    """
    設定 raw_config を正規化・展開・検証して dict を返す。
    output_path を None にするとファイル出力しない。
    """
    logger.info("設定データの読み込み開始")
    try:
        raw = deepcopy(raw_config)

        sim_config =  _parse_simulation(raw)
        node_config = _parse_nodes(raw)
        ventilation_config = _parse_chain_branches(raw, "ventilation_branches")
        thermal_config = _parse_chain_branches(raw, "thermal_branches")
        surface_config = _parse_surface(raw)
        aircon_config = _parse_aircon(raw)

        logger.info("計算フラグの自動設定を開始します")
        for flag in ("p", "t", "x", "c"):
            has_flag = any(
                isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
                for node in node_config
            )
            sim_config["calc_flag"][flag] = has_flag

        logger.info("設定データの処理が完了しました")

        if surface_config:
            logger.info("表面の解析を開始します。")
            surface_data = surface_config
            for surface in surface_config:
                add_nodes, add_thermal_branches = process_surface(surface)
                node_config.extend(add_nodes)
                thermal_config.extend(add_thermal_branches)
            logger.info("表面の解析が完了しました。")

            logger.info("日射の解析を開始します。")
            sim_length = int(sim_config["index"]["length"])
            for surface in surface_data:
                if "solar" in surface:
                    if surface["part"] in ["wall", "floor", "ceiling"]:
                        add_thermal_branches = process_wall_solar(surface, sim_length)
                    elif surface["part"] == "glass":
                        add_thermal_branches = process_glass_solar(surface, surface_data, sim_length)
                    thermal_config.extend(add_thermal_branches)
            logger.info("日射の解析が完了しました。")

            logger.info("室内放射の解析を開始します。")
            node_with_surface = [s["key"].split(CHAIN_DELIMITER)[0] for s in surface_data]
            for node in list(set(node_with_surface)):
                surfaces = [s for s in surface_data if s["key"].startswith(node)]
                add_thermal_branches = process_radiation(node, surfaces)
                thermal_config.extend(add_thermal_branches)
            logger.info("室内放射の解析が完了しました。")

        if "aircon" in raw_config:
            logger.info("空調の解析を開始します。")
            for aircon in raw_config["aircon"]:
                add_nodes, add_ventilation_branches = process_aircon(aircon)
                node_config.extend(add_nodes)
                ventilation_config.extend(add_ventilation_branches)
            logger.info("空調の解析が完了しました。")

        for node in node_config:
            logger.info("熱容量を追加します")
            if "thermal_mass" in node:
                add_nodes, add_thermal_branches = add_capacity(node, sim_config["index"]["timestep"])
                node_config.extend(add_nodes)
                thermal_config.extend(add_thermal_branches)
            logger.info("熱容量の追加が完了しました。")

        # 設定ファイルの保存（前に検証を実施）
        output_json = {
            "simulation": sim_config,
            "nodes": node_config,
            "ventilation_branches": ventilation_config,
            "thermal_branches": thermal_config,
            "aircon": aircon_config,
        }

        # 出力のバリデーション（辞書入力で検証・正規化）
        output_json = validate_dict(output_json)

        # JSON 出力（必要に応じて）
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)
            logger.info(f"設定データを {output_path} に出力しました")

        return output_json

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise



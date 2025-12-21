from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy
import gzip
import json

from .logger import get_logger
from .parsers import parse_all
from .surfaces import process_surfaces
from .aircon import process_aircons
from .thermal import process_capacities
from .validate import validate_dict

logger = get_logger(__name__)


# ------------------------------
# エントリポイント
# ------------------------------
def build_config(
    raw_config: Dict[str, Any],
    output_path: Optional[str] = "parsed_input_data.json",
    add_surface: bool = True,
    add_aircon: bool = True,
    add_capacity: bool = True,
    add_surface_solar: bool = True,
    add_surface_radiation: bool = True,
) -> Dict[str, Any]:
    """
    設定 raw_config を正規化・展開・検証して dict を返す。
    output_path を None にするとファイル出力しない。
    add_surface / add_aircon / add_capacity で各処理の有無を制御できる。
    add_surface_solar / add_surface_radiation で表面の日射・室内放射処理を個別に制御できる。
    """
    logger.info("設定データの読み込み開始")
    try:
        raw = deepcopy(raw_config)

        # 設定データの解析
        sim_config, node_config, ventilation_config, thermal_config, surface_config, aircon_config = parse_all(raw)

        # 表面の処理
        if surface_config and add_surface:
            sim_length = int(sim_config["index"]["length"])
            add_nodes, add_tb = process_surfaces(
                surface_config,
                sim_length,
                add_solar=add_surface_solar,
                add_radiation=add_surface_radiation,
            )
            node_config.extend(add_nodes)
            thermal_config.extend(add_tb)
        elif surface_config:
            logger.info("表面の処理をスキップします。")

        # 空調の処理
        if aircon_config and add_aircon:
            add_nodes, add_ventilation_branches = process_aircons(aircon_config)
            node_config.extend(add_nodes)
            ventilation_config.extend(add_ventilation_branches)
        elif aircon_config:
            logger.info("空調の処理をスキップします。")
        
        # 熱容量の処理
        if add_capacity:
            add_nodes, add_thermal_branches = process_capacities(node_config, sim_config["index"]["timestep"])
            node_config.extend(add_nodes)
            thermal_config.extend(add_thermal_branches)
        else:
            logger.info("熱容量の処理をスキップします。")

        # 計算フラグの自動設定
        logger.info("計算フラグの自動設定を開始します")
        for flag in ("p", "t", "x", "c"):
            has_flag = any(
                isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
                for node in node_config
            )
            sim_config["calc_flag"][flag] = has_flag

        # 設定ファイルの保存
        output_json = {
            "simulation": sim_config,
            "nodes": node_config,
            "ventilation_branches": ventilation_config,
            "thermal_branches": thermal_config,
            "aircon": aircon_config,
        }

        # 出力のバリデーション
        output_json = validate_dict(output_json)

        # JSON 出力
        if output_path:
            # 大規模JSONは同じ数列が大量に繰り返されるため gzip 圧縮が非常に効く。
            # output_path が *.gz の場合は圧縮して保存する。
            if str(output_path).lower().endswith(".gz"):
                with gzip.open(output_path, "wt", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, separators=(",", ":"))
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=4, ensure_ascii=False)
            logger.info(f"設定データを {output_path} に出力しました")

        return output_json

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise



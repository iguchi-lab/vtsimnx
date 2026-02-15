from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy
import gzip
import json

from .logger import get_logger
from .parsers import parse_all
from .surfaces import process_surfaces
from .heat_sources import build_heat_generation_branches
from .moisture import build_humidity_generation_vents
from .aircon import process_aircons
from .thermal import process_capacities
from .validate import validate_dict, validate_dict_with_warnings, validate_dict_with_warning_details

logger = get_logger(__name__)


# ------------------------------
# エントリポイント
# ------------------------------
def build_config_with_warnings(
    raw_config: Dict[str, Any],
    # output_path を指定しない場合はファイルを出力しない（容量節約）
    output_path: Optional[str] = None,
    add_surface: bool | None = None,
    add_aircon: bool | None = None,
    add_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    surface_layer_method: str = "rc",
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> tuple[Dict[str, Any], list[str]]:
    """
    設定 raw_config を正規化・展開・検証して dict と warnings を返す。
    output_path を None にするとファイル出力しない。
    add_surface / add_aircon / add_capacity で各処理の有無を制御できる。
    add_surface_solar / add_surface_radiation で表面の日射・室内放射処理を個別に制御できる。
    """
    logger.info("設定データの読み込み開始")
    try:
        raw = deepcopy(raw_config)

        # builder の ON/OFF オプションは raw_config["builder"] でも指定できる。
        # API など外部から明示指定された引数（None 以外）が最優先。
        builder_opt = raw.get("builder")
        if isinstance(builder_opt, dict):
            def _pick_bool(name: str) -> bool | None:
                v = builder_opt.get(name)
                return v if isinstance(v, bool) else None

            if add_surface is None:
                add_surface = _pick_bool("add_surface")
            if add_aircon is None:
                add_aircon = _pick_bool("add_aircon")
            if add_capacity is None:
                add_capacity = _pick_bool("add_capacity")
            if add_surface_solar is None:
                add_surface_solar = _pick_bool("add_surface_solar")
            if add_surface_nocturnal is None:
                add_surface_nocturnal = _pick_bool("add_surface_nocturnal")
            if add_surface_radiation is None:
                add_surface_radiation = _pick_bool("add_surface_radiation")

        # 互換: トップレベルに置くのも許可（builder より優先度は低い）
        if add_surface is None and isinstance(raw.get("add_surface"), bool):
            add_surface = raw.get("add_surface")
        if add_aircon is None and isinstance(raw.get("add_aircon"), bool):
            add_aircon = raw.get("add_aircon")
        if add_capacity is None and isinstance(raw.get("add_capacity"), bool):
            add_capacity = raw.get("add_capacity")
        if add_surface_solar is None and isinstance(raw.get("add_surface_solar"), bool):
            add_surface_solar = raw.get("add_surface_solar")
        if add_surface_nocturnal is None and isinstance(raw.get("add_surface_nocturnal"), bool):
            add_surface_nocturnal = raw.get("add_surface_nocturnal")
        if add_surface_radiation is None and isinstance(raw.get("add_surface_radiation"), bool):
            add_surface_radiation = raw.get("add_surface_radiation")

        # 最終デフォルト（従来互換: 指定が無ければ全て True）
        add_surface = True if add_surface is None else bool(add_surface)
        add_aircon = True if add_aircon is None else bool(add_aircon)
        add_capacity = True if add_capacity is None else bool(add_capacity)
        add_surface_solar = True if add_surface_solar is None else bool(add_surface_solar)
        add_surface_nocturnal = True if add_surface_nocturnal is None else bool(add_surface_nocturnal)
        add_surface_radiation = True if add_surface_radiation is None else bool(add_surface_radiation)

        # JSON から builder オプションを読み取る（関数引数が既定値のときだけ反映）
        # 例:
        #   "builder": { "surface_layer_method": "response" }
        # 互換: トップレベルに "surface_layer_method" を置くのも許可
        if surface_layer_method == "rc":
            if isinstance(builder_opt, dict):
                v = builder_opt.get("surface_layer_method")
                if isinstance(v, str) and v:
                    surface_layer_method = v
                rm = builder_opt.get("response_method")
                if response_method == "arx_rc" and isinstance(rm, str) and rm:
                    response_method = rm
                rt = builder_opt.get("response_terms")
                if response_terms is None and rt is not None:
                    try:
                        response_terms = int(rt)
                    except Exception:
                        raise ValueError(f"builder.response_terms must be int, got {rt!r}")
            v2 = raw.get("surface_layer_method")
            if isinstance(v2, str) and v2:
                surface_layer_method = v2
            # 互換: トップレベルでも受ける
            if response_method == "arx_rc":
                rm2 = raw.get("response_method")
                if isinstance(rm2, str) and rm2:
                    response_method = rm2
            if response_terms is None:
                rt2 = raw.get("response_terms")
                if rt2 is not None:
                    try:
                        response_terms = int(rt2)
                    except Exception:
                        raise ValueError(f"response_terms must be int, got {rt2!r}")

        # 設定データの解析
        sim_config, node_config, ventilation_config, thermal_config, surface_config, aircon_config = parse_all(raw)

        # 表面の処理
        if surface_config and add_surface:
            sim_length = int(sim_config["index"]["length"])
            add_nodes, add_tb = process_surfaces(
                surface_config,
                sim_length,
                node_config=node_config,
                add_solar=add_surface_solar,
                add_nocturnal=add_surface_nocturnal,
                add_radiation=add_surface_radiation,
                layer_method=surface_layer_method,
                time_step=float(sim_config["index"]["timestep"]),
                response_method=response_method,
                response_terms=response_terms,
            )
            node_config.extend(add_nodes)
            thermal_config.extend(add_tb)
        elif surface_config:
            logger.info("表面の処理をスキップします。")

        # 発熱（heat_source）の処理
        # - 対流分: void->room の heat_generation
        # - 放射分: void->surface の heat_generation（面積按分; surfacesが無ければ void->room）
        try:
            thermal_config.extend(build_heat_generation_branches(raw_config=raw, surface_config=surface_config))
        except Exception as e:
            logger.exception("heat_source の処理に失敗しました: %s", e)
            raise

        # 発湿（humidity_source）の処理
        # - 換気ブランチに humidity_generation を付与して solver 側に渡す（空気移動は vol=0.0）
        try:
            add_vents, rooms = build_humidity_generation_vents(raw_config=raw)
            if add_vents:
                ventilation_config.extend(add_vents)
                # 発湿が指定された室は、calc_x を自動でON（ユーザーが忘れても湿気計算を有効化できる）
                room_set = set(str(r) for r in rooms)
                for node in node_config:
                    if isinstance(node, dict) and str(node.get("key", "")) in room_set:
                        node["calc_x"] = True
        except Exception as e:
            logger.exception("humidity_source の処理に失敗しました: %s", e)
            raise

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

        # 出力のバリデーション（warnings も返す）
        validated, warnings = validate_dict_with_warnings(output_json)

        # JSON 出力
        if output_path:
            # 大規模JSONは同じ数列が大量に繰り返されるため gzip 圧縮が非常に効く。
            # output_path が *.gz の場合は圧縮して保存する。
            if str(output_path).lower().endswith(".gz"):
                with gzip.open(output_path, "wt", encoding="utf-8") as f:
                    json.dump(validated, f, ensure_ascii=False, separators=(",", ":"))
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(validated, f, indent=4, ensure_ascii=False)
            logger.info(f"設定データを {output_path} に出力しました")

        return validated, warnings

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise


def build_config_with_warning_details(
    raw_config: Dict[str, Any],
    # output_path を指定しない場合はファイルを出力しない（容量節約）
    output_path: Optional[str] = None,
    add_surface: bool | None = None,
    add_aircon: bool | None = None,
    add_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    surface_layer_method: str = "rc",
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> tuple[Dict[str, Any], list[str], list[dict]]:
    """
    設定 raw_config を正規化・展開・検証して dict と warnings（文字列/構造化）を返す。
    """
    logger.info("設定データの読み込み開始")
    raw = deepcopy(raw_config)

    # builder の ON/OFF オプションは raw_config["builder"] でも指定できる。
    # API など外部から明示指定された引数（None 以外）が最優先。
    builder_opt = raw.get("builder")
    if isinstance(builder_opt, dict):
        def _pick_bool(name: str) -> bool | None:
            v = builder_opt.get(name)
            return v if isinstance(v, bool) else None

        if add_surface is None:
            add_surface = _pick_bool("add_surface")
        if add_aircon is None:
            add_aircon = _pick_bool("add_aircon")
        if add_capacity is None:
            add_capacity = _pick_bool("add_capacity")
        if add_surface_solar is None:
            add_surface_solar = _pick_bool("add_surface_solar")
        if add_surface_nocturnal is None:
            add_surface_nocturnal = _pick_bool("add_surface_nocturnal")
        if add_surface_radiation is None:
            add_surface_radiation = _pick_bool("add_surface_radiation")

    # 互換: トップレベルに置くのも許可（builder より優先度は低い）
    if add_surface is None and isinstance(raw.get("add_surface"), bool):
        add_surface = raw.get("add_surface")
    if add_aircon is None and isinstance(raw.get("add_aircon"), bool):
        add_aircon = raw.get("add_aircon")
    if add_capacity is None and isinstance(raw.get("add_capacity"), bool):
        add_capacity = raw.get("add_capacity")
    if add_surface_solar is None and isinstance(raw.get("add_surface_solar"), bool):
        add_surface_solar = raw.get("add_surface_solar")
    if add_surface_nocturnal is None and isinstance(raw.get("add_surface_nocturnal"), bool):
        add_surface_nocturnal = raw.get("add_surface_nocturnal")
    if add_surface_radiation is None and isinstance(raw.get("add_surface_radiation"), bool):
        add_surface_radiation = raw.get("add_surface_radiation")

    # 最終デフォルト（従来互換: 指定が無ければ全て True）
    add_surface = True if add_surface is None else bool(add_surface)
    add_aircon = True if add_aircon is None else bool(add_aircon)
    add_capacity = True if add_capacity is None else bool(add_capacity)
    add_surface_solar = True if add_surface_solar is None else bool(add_surface_solar)
    add_surface_nocturnal = True if add_surface_nocturnal is None else bool(add_surface_nocturnal)
    add_surface_radiation = True if add_surface_radiation is None else bool(add_surface_radiation)

    # JSON から builder オプションを読み取る（関数引数が既定値のときだけ反映）
    if surface_layer_method == "rc":
        if isinstance(builder_opt, dict):
            v = builder_opt.get("surface_layer_method")
            if isinstance(v, str) and v:
                surface_layer_method = v
            rm = builder_opt.get("response_method")
            if response_method == "arx_rc" and isinstance(rm, str) and rm:
                response_method = rm
            rt = builder_opt.get("response_terms")
            if response_terms is None and rt is not None:
                try:
                    response_terms = int(rt)
                except Exception:
                    raise ValueError(f"builder.response_terms must be int, got {rt!r}")
        v2 = raw.get("surface_layer_method")
        if isinstance(v2, str) and v2:
            surface_layer_method = v2
        if response_method == "arx_rc":
            rm2 = raw.get("response_method")
            if isinstance(rm2, str) and rm2:
                response_method = rm2
        if response_terms is None:
            rt2 = raw.get("response_terms")
            if rt2 is not None:
                try:
                    response_terms = int(rt2)
                except Exception:
                    raise ValueError(f"response_terms must be int, got {rt2!r}")

    logger.info("設定データのパース開始: keys=%d", len(raw) if isinstance(raw, dict) else -1)
    sim_config, node_config, ventilation_config, thermal_config, surface_config, aircon_config = parse_all(raw)
    logger.info(
        "設定データのパース完了: nodes=%d, vents=%d, thermals=%d, surfaces=%d, aircons=%d",
        len(node_config) if node_config is not None else -1,
        len(ventilation_config) if ventilation_config is not None else -1,
        len(thermal_config) if thermal_config is not None else -1,
        len(surface_config) if surface_config is not None else -1,
        len(aircon_config) if aircon_config is not None else -1,
    )

    if surface_config and add_surface:
        sim_length = int(sim_config["index"]["length"])
        add_nodes, add_tb = process_surfaces(
            surface_config,
            sim_length,
            node_config=node_config,
            add_solar=add_surface_solar,
            add_nocturnal=add_surface_nocturnal,
            add_radiation=add_surface_radiation,
            layer_method=surface_layer_method,
            time_step=float(sim_config["index"]["timestep"]),
            response_method=response_method,
            response_terms=response_terms,
        )
        node_config.extend(add_nodes)
        thermal_config.extend(add_tb)
    elif surface_config:
        logger.info("表面の処理をスキップします。")

    # 発熱（heat_source）の処理
    try:
        thermal_config.extend(build_heat_generation_branches(raw_config=raw, surface_config=surface_config))
    except Exception as e:
        logger.exception("heat_source の処理に失敗しました: %s", e)
        raise

    if aircon_config and add_aircon:
        add_nodes, add_ventilation_branches = process_aircons(aircon_config)
        node_config.extend(add_nodes)
        ventilation_config.extend(add_ventilation_branches)
    elif aircon_config:
        logger.info("空調の処理をスキップします。")

    if add_capacity:
        add_nodes, add_thermal_branches = process_capacities(node_config, sim_config["index"]["timestep"])
        node_config.extend(add_nodes)
        thermal_config.extend(add_thermal_branches)
    else:
        logger.info("熱容量の処理をスキップします。")

    logger.info("計算フラグの自動設定を開始します")
    for flag in ("p", "t", "x", "c"):
        has_flag = any(
            isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
            for node in node_config
        )
        sim_config["calc_flag"][flag] = has_flag

    output_json = {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
        "aircon": aircon_config,
    }

    validated, warnings, warning_details = validate_dict_with_warning_details(output_json)

    if output_path:
        if str(output_path).lower().endswith(".gz"):
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(validated, f, ensure_ascii=False, separators=(",", ":"))
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(validated, f, indent=4, ensure_ascii=False)
        logger.info(f"設定データを {output_path} に出力しました")

    return validated, warnings, warning_details


def build_config(
    raw_config: Dict[str, Any],
    # output_path を指定しない場合はファイルを出力しない（容量節約）
    output_path: Optional[str] = None,
    add_surface: bool | None = None,
    add_aircon: bool | None = None,
    add_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    surface_layer_method: str = "rc",
    response_method: str = "arx_rc",
    response_terms: int | None = None,
) -> Dict[str, Any]:
    """
    設定 raw_config を正規化・展開・検証して dict を返す。
    output_path を None にするとファイル出力しない。
    add_surface / add_aircon / add_capacity で各処理の有無を制御できる。
    add_surface_solar / add_surface_radiation で表面の日射・室内放射処理を個別に制御できる。
    """
    validated, _warnings = build_config_with_warnings(
        raw_config,
        output_path=output_path,
        add_surface=add_surface,
        add_aircon=add_aircon,
        add_capacity=add_capacity,
        add_surface_solar=add_surface_solar,
        add_surface_nocturnal=add_surface_nocturnal,
        add_surface_radiation=add_surface_radiation,
        surface_layer_method=surface_layer_method,
        response_method=response_method,
        response_terms=response_terms,
    )
    return validated



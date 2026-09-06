from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy
import gzip
import json
import time

from .build_options import BuildOptions
from .logger import get_logger
from .parsers import parse_all
from .surfaces import process_surfaces
from .heat_sources import build_heat_generation_branches
from .moisture import build_humidity_generation_vents
from .aircon import process_aircons
from .thermal import process_capacities
from .moisture_capacity import (
    derive_calc_x_from_moisture_capacity,
    process_moisture_capacities,
    strip_moisture_capacity_fields,
)
from .validate import validate_dict_with_warnings, validate_dict_with_warning_details

logger = get_logger(__name__)


def _build_output_json(raw: Dict[str, Any], *, options: BuildOptions) -> Dict[str, Any]:
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

    if surface_config and options.add_surface:
        sim_length = int(sim_config["index"]["length"])
        add_nodes, add_tb = process_surfaces(
            surface_config,
            sim_length,
            node_config=node_config,
            add_solar=options.add_surface_solar,
            add_nocturnal=options.add_surface_nocturnal,
            add_radiation=options.add_surface_radiation,
            radiation_exclude_glass=options.add_surface_radiation_exclude_glass,
            layer_method=options.surface_layer_method,
            time_step=float(sim_config["index"]["timestep"]),
            response_method=options.response_method,
            response_terms=options.response_terms,
        )
        node_config.extend(add_nodes)
        thermal_config.extend(add_tb)
    elif surface_config:
        logger.info("表面の処理をスキップします。")

    try:
        thermal_config.extend(build_heat_generation_branches(raw_config=raw, surface_config=surface_config))
    except Exception as e:
        logger.exception("heat_source の処理に失敗しました: %s", e)
        raise

    try:
        add_vents, rooms = build_humidity_generation_vents(raw_config=raw)
        if add_vents:
            ventilation_config.extend(add_vents)
            room_set = set(str(r) for r in rooms)
            for node in node_config:
                if isinstance(node, dict) and str(node.get("key", "")) in room_set:
                    node["calc_x"] = True
    except Exception as e:
        logger.exception("humidity_source の処理に失敗しました: %s", e)
        raise

    # 湿気容量: True なら展開前に calc_x を立てる。False ならフィールド除去（無効化）。
    if options.add_moisture_capacity:
        derive_calc_x_from_moisture_capacity(node_config)
    else:
        strip_moisture_capacity_fields(node_config)

    if aircon_config and options.add_aircon:
        # 濃度: set が calc_c なら aircon にも引き継ぐ（濃度ソルバの未知数）。
        # 湿度: aircon は常に calc_x=false。吹出絶対湿度 supplyX（運転中）/
        # 吸込パススルー（OFF）を固定境界として current_x に書くため。
        # 以前の calc_x 伝播は湿度ソルバが supplyX を上書きし、外側ループが
        # SupplyHumidityChanged で振動する原因になった。
        calc_c_node_keys = {
            str(node.get("key", ""))
            for node in node_config
            if isinstance(node, dict) and bool(node.get("calc_c", False))
        }
        for ac in aircon_config:
            if not isinstance(ac, dict):
                continue
            set_node = ac.get("set", ac.get("in"))
            key = str(set_node)
            ac["calc_x"] = False
            ac["calc_c"] = key in calc_c_node_keys

        add_nodes, add_ventilation_branches = process_aircons(aircon_config)
        node_config.extend(add_nodes)
        ventilation_config.extend(add_ventilation_branches)
    elif aircon_config:
        logger.info("空調の処理をスキップします。")

    if options.add_capacity:
        add_nodes, add_thermal_branches = process_capacities(node_config, sim_config["index"]["timestep"])
        node_config.extend(add_nodes)
        thermal_config.extend(add_thermal_branches)
    else:
        logger.info("熱容量の処理をスキップします。")

    if options.add_moisture_capacity:
        add_nodes, add_thermal_branches = process_moisture_capacities(node_config, sim_config["index"]["timestep"])
        node_config.extend(add_nodes)
        thermal_config.extend(add_thermal_branches)
    else:
        logger.info("湿気容量の処理をスキップします。")

    logger.info("計算フラグの自動設定を開始します")
    for flag in ("p", "t", "x", "c"):
        has_flag = any(
            isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
            for node in node_config
        )
        sim_config["calc_flag"][flag] = has_flag

    return {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
        "aircon": aircon_config,
    }


def _write_output_json_if_needed(validated: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    if str(output_path).lower().endswith(".gz"):
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(validated, f, ensure_ascii=False, separators=(",", ":"))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=4, ensure_ascii=False)
    logger.info(f"設定データを {output_path} に出力しました")


def _build_core(
    raw_config: Dict[str, Any],
    *,
    output_path: Optional[str],
    add_surface: bool | None,
    add_aircon: bool | None,
    add_capacity: bool | None,
    add_moisture_capacity: bool | None,
    add_surface_solar: bool | None,
    add_surface_nocturnal: bool | None,
    add_surface_radiation: bool | None,
    add_surface_radiation_exclude_glass: bool | None,
    surface_layer_method: str | None,
    response_method: str | None,
    response_terms: int | None,
    with_warning_details: bool,
    build_stats_out: Optional[list] = None,
    unknown_keys: str = "strip",
) -> tuple[Dict[str, Any], list[str], list[dict] | None]:
    from app.builder.validate import use_unknown_keys_mode

    start = time.perf_counter()
    build_counts: tuple[int, int, int] | None = None  # (nodes, thermal_branches, ventilation_branches)
    try:
        raw = deepcopy(raw_config)
        options = BuildOptions.resolve(
            raw,
            add_surface=add_surface,
            add_aircon=add_aircon,
            add_capacity=add_capacity,
            add_moisture_capacity=add_moisture_capacity,
            add_surface_solar=add_surface_solar,
            add_surface_nocturnal=add_surface_nocturnal,
            add_surface_radiation=add_surface_radiation,
            add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
            surface_layer_method=surface_layer_method,
            response_method=response_method,
            response_terms=response_terms,
        )

        output_json = _build_output_json(raw, options=options)
        build_counts = (
            len(output_json.get("nodes") or []),
            len(output_json.get("thermal_branches") or []),
            len(output_json.get("ventilation_branches") or []),
        )

        mode = "error" if unknown_keys == "error" else "strip"
        with use_unknown_keys_mode(mode):  # type: ignore[arg-type]
            if with_warning_details:
                validated, warnings, warning_details = validate_dict_with_warning_details(output_json)
                _write_output_json_if_needed(validated, output_path)
                return validated, warnings, warning_details

            validated, warnings = validate_dict_with_warnings(output_json)
            _write_output_json_if_needed(validated, output_path)
            return validated, warnings, None
    finally:
        elapsed = time.perf_counter() - start
        if build_counts is not None and build_stats_out is not None:
            build_stats_out.append(build_counts)
        logger.info("ビルダー所要時間: %.3f 秒", elapsed)


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
    add_moisture_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    add_surface_radiation_exclude_glass: bool | None = None,
    surface_layer_method: str | None = None,
    response_method: str | None = None,
    response_terms: int | None = None,
    build_stats_out: Optional[list] = None,
    unknown_keys: str = "strip",
) -> tuple[Dict[str, Any], list[str]]:
    """
    設定 raw_config を正規化・展開・検証して dict と warnings を返す。
    output_path を None にするとファイル出力しない。
    add_surface / add_aircon / add_capacity で各処理の有無を制御できる。
    add_surface_solar / add_surface_radiation で表面の日射・室内放射処理を個別に制御できる。
    add_surface_radiation_exclude_glass で室内放射対象からガラス面を除外できる。
    surface_layer_method / response_method は None（未指定）のとき JSON / 既定値で解決する。
    """
    logger.info("設定データの読み込み開始")
    try:
        validated, warnings, _details = _build_core(
            raw_config,
            output_path=output_path,
            add_surface=add_surface,
            add_aircon=add_aircon,
            add_capacity=add_capacity,
            add_moisture_capacity=add_moisture_capacity,
            add_surface_solar=add_surface_solar,
            add_surface_nocturnal=add_surface_nocturnal,
            add_surface_radiation=add_surface_radiation,
            add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
            surface_layer_method=surface_layer_method,
            response_method=response_method,
            response_terms=response_terms,
            with_warning_details=False,
            build_stats_out=build_stats_out,
            unknown_keys=unknown_keys,
        )
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
    add_moisture_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    add_surface_radiation_exclude_glass: bool | None = None,
    surface_layer_method: str | None = None,
    response_method: str | None = None,
    response_terms: int | None = None,
    build_stats_out: Optional[list] = None,
    unknown_keys: str = "strip",
) -> tuple[Dict[str, Any], list[str], list[dict]]:
    """
    設定 raw_config を正規化・展開・検証して dict と warnings（文字列/構造化）を返す。
    surface_layer_method / response_method は None（未指定）のとき JSON / 既定値で解決する。
    """
    logger.info("設定データの読み込み開始")
    validated, warnings, warning_details = _build_core(
        raw_config,
        output_path=output_path,
        add_surface=add_surface,
        add_aircon=add_aircon,
        add_capacity=add_capacity,
        add_moisture_capacity=add_moisture_capacity,
        add_surface_solar=add_surface_solar,
        add_surface_nocturnal=add_surface_nocturnal,
        add_surface_radiation=add_surface_radiation,
        add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
        surface_layer_method=surface_layer_method,
        response_method=response_method,
        response_terms=response_terms,
        with_warning_details=True,
        build_stats_out=build_stats_out,
        unknown_keys=unknown_keys,
    )
    assert warning_details is not None
    return validated, warnings, warning_details


def build_config(
    raw_config: Dict[str, Any],
    # output_path を指定しない場合はファイルを出力しない（容量節約）
    output_path: Optional[str] = None,
    add_surface: bool | None = None,
    add_aircon: bool | None = None,
    add_capacity: bool | None = None,
    add_moisture_capacity: bool | None = None,
    add_surface_solar: bool | None = None,
    add_surface_nocturnal: bool | None = None,
    add_surface_radiation: bool | None = None,
    add_surface_radiation_exclude_glass: bool | None = None,
    surface_layer_method: str | None = None,
    response_method: str | None = None,
    response_terms: int | None = None,
) -> Dict[str, Any]:
    """
    設定 raw_config を正規化・展開・検証して dict を返す。
    output_path を None にするとファイル出力しない。
    add_surface / add_aircon / add_capacity で各処理の有無を制御できる。
    add_surface_solar / add_surface_radiation で表面の日射・室内放射処理を個別に制御できる。
    add_surface_radiation_exclude_glass で室内放射対象からガラス面を除外できる。
    surface_layer_method / response_method は None（未指定）のとき JSON / 既定値で解決する。
    """
    validated, _warnings = build_config_with_warnings(
        raw_config,
        output_path=output_path,
        add_surface=add_surface,
        add_aircon=add_aircon,
        add_capacity=add_capacity,
        add_moisture_capacity=add_moisture_capacity,
        add_surface_solar=add_surface_solar,
        add_surface_nocturnal=add_surface_nocturnal,
        add_surface_radiation=add_surface_radiation,
        add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
        surface_layer_method=surface_layer_method,
        response_method=response_method,
        response_terms=response_terms,
    )
    return validated

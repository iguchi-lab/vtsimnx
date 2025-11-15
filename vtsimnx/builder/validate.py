from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import numpy as np

from .logger import get_logger
from .utils import CHAIN_DELIMITER, convert_numeric_values, convert_to_json_compatible
from .config_types import (
    SimConfigType,
    NodeType,
    VentilationBranchType,
    ThermalBranchType,
    NodeTypeEnum,
    VentilationBranchTypeEnum,
    ThermalBranchTypeEnum,
)

logger = get_logger(__name__)


# ------------------------------
# 例外と結果型
# ------------------------------
class ValidationError(Exception):
    pass


class ConfigFileError(ValidationError):
    pass


class NodeError(ValidationError):
    pass


class BranchError(ValidationError):
    pass


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]


# ------------------------------
# 共通バリデーション
# ------------------------------
def _allowed_keys(typed_dict_cls) -> Set[str]:
    """TypedDict から許可キー集合を取得する。"""
    return set(getattr(typed_dict_cls, "__annotations__", {}).keys())


def log_and_strip_unknown_fields(data: Dict[str, Any], allowed: Set[str], context: str) -> None:
    """
    config_types.py に無いキーがあればログに警告を出し、取り除く。
    - 目的: タイプミスを早期に発見しつつ、処理は継続する
    """
    unknown = [k for k in list(data.keys()) if k not in allowed]
    for k in unknown:
        logger.warning("%s に未定義のフィールド '%s' が指定されました。無視しました。", context, k)
        data.pop(k, None)


def validate_required_fields(data: Dict[str, Any], required_fields: List[str], context: str) -> ValidationResult:
    errors: List[str] = []
    for field in required_fields:
        if field not in data:
            errors.append(f"{context}に'{field}'フィールドがありません")
    return ValidationResult(len(errors) == 0, errors, [])


def validate_node_chain(key: str, source: str, target: str, context: str) -> ValidationResult:
    if source == target:
        return ValidationResult(False, [f"{context} {key} の'source'ノードと'target'ノードが同じです"], [])
    return ValidationResult(True, [], [])


def validate_node_exists(node_key: str, node_config: List[NodeType], context: str) -> ValidationResult:
    if not any(n.get("key") == node_key for n in node_config):
        return ValidationResult(False, [f"{context}のノード '{node_key}' が存在しません"], [])
    return ValidationResult(True, [], [])


def validate_branch_type(branch: Dict[str, Any], branch_types: Dict[str, Dict[str, List[str]]], context: str) -> ValidationResult:
    errors: List[str] = []
    if "type" not in branch:
        for type_name, requirements in branch_types.items():
            if all(param in branch for param in requirements.get("required", [])):
                branch["type"] = type_name
                break
        else:
            errors.append(f"{context} {branch.get('key', '?')} の'type'が不正です")

    if branch.get("type") not in branch_types:
        errors.append(f"{context} {branch.get('key', '?')} の'type'が不正です")

    return ValidationResult(len(errors) == 0, errors, [])


def validate_branch_parameters(branch: Dict[str, Any], branch_types: Dict[str, Dict[str, List[str]]], context: str) -> ValidationResult:
    errors: List[str] = []
    required = branch_types[branch["type"]].get("required", [])
    for param in required:
        if param not in branch:
            errors.append(f"{context} {branch['key']} に'{param}'が指定されていません")
    return ValidationResult(len(errors) == 0, errors, [])


def set_default_generation(branch: Dict[str, Any], sim_config: SimConfigType, field: str, calc_flag: str) -> None:
    if sim_config["calc_flag"][calc_flag]:
        branch[field] = branch.get(field, np.zeros(sim_config["index"]["length"]))


# ------------------------------
# 設定ファイルのバリデーション
# ------------------------------
def validate_config_file(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        raw_config = convert_numeric_values(raw_config)

        # 最低限の必須セクション
        for section in ["simulation", "nodes", "ventilation_branches", "thermal_branches"]:
            if section not in raw_config:
                raise ConfigFileError(f"設定ファイルに'{section}'セクションがありません")
        return raw_config
    except json.JSONDecodeError as e:
        raise ConfigFileError(f"JSONファイルの解析に失敗しました: {e}")
    except Exception as e:
        raise ConfigFileError(f"設定ファイルの読み込みに失敗しました: {e}")


# ------------------------------
# 各セクションのバリデーション
# ------------------------------
def validate_sim_config(sim_config: Dict[str, Any]) -> Tuple[SimConfigType, ValidationResult]:
    logger.info("sim_configのバリデーションを開始します。")

    result = validate_required_fields(sim_config, ["index", "tolerance", "calc_flag"], "sim_config")
    if not result.is_valid:
        return sim_config, result

    # 未知キーはログに出しつつ削除
    log_and_strip_unknown_fields(sim_config, _allowed_keys(SimConfigType), "sim_config")

    result = validate_required_fields(sim_config["index"], ["start", "end", "timestep", "length"], "indexセクション")
    if not result.is_valid:
        return sim_config, result

    # サブセクションの未知キー除去
    from .config_types import IndexType, ToleranceType, CalcFlagType  # 局所 import で循環回避の保険
    log_and_strip_unknown_fields(sim_config["index"], _allowed_keys(IndexType), "indexセクション")
    log_and_strip_unknown_fields(sim_config["tolerance"], _allowed_keys(ToleranceType), "toleranceセクション")
    log_and_strip_unknown_fields(sim_config["calc_flag"], _allowed_keys(CalcFlagType), "calc_flagセクション")

    try:
        datetime.fromisoformat(str(sim_config["index"]["start"]).replace("Z", "+00:00"))
        datetime.fromisoformat(str(sim_config["index"]["end"]).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return sim_config, ValidationResult(False, ["startまたはendの日時形式が不正です"], [])

    result = validate_required_fields(
        sim_config["tolerance"], ["ventilation", "thermal", "convergence"], "toleranceセクション"
    )
    if not result.is_valid:
        return sim_config, result

    result = validate_required_fields(sim_config["calc_flag"], ["p", "t", "x", "c"], "calc_flagセクション")
    if not result.is_valid:
        return sim_config, result

    logger.info("sim_configのバリデーションが完了しました。")
    return sim_config, ValidationResult(True, [], [])


def validate_node_config(
    sim_config: SimConfigType,
    node_config: List[Dict[str, Any]],
    ventilation_config: List[Dict[str, Any]],
    thermal_config: List[Dict[str, Any]],
) -> Tuple[List[NodeType], ValidationResult]:
    logger.info("node_configのバリデーションを開始します。")

    if not isinstance(node_config, list):
        return node_config, ValidationResult(False, ["node_configはリストである必要があります"], [])

    errors: List[str] = []
    warnings: List[str] = []

    # 換気・熱で使われるノード集合を抽出
    ventilation_nodes = set()
    for branch in ventilation_config:
        parts = str(branch["key"]).split(CHAIN_DELIMITER)
        if len(parts) == 2:
            ventilation_nodes.add(parts[0])
            ventilation_nodes.add(parts[1])

    thermal_nodes = set()
    for branch in thermal_config:
        parts = str(branch["key"]).split(CHAIN_DELIMITER)
        if len(parts) == 2:
            thermal_nodes.add(parts[0])
            thermal_nodes.add(parts[1])

    for node in node_config:
        result = validate_required_fields(node, ["key"], "ノード")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # 未知キーはログ出力して削除（例: thermal_mass などはここで落とす）
        log_and_strip_unknown_fields(node, _allowed_keys(NodeType), f"ノード {node.get('key', '?')}")

        # ノードタイプの既定値
        node["type"] = node.get("type", NodeTypeEnum.NORMAL)

        # p の既定値・形式（換気に関与するノードのみ）
        if node["key"] in ventilation_nodes:
            p_val = node.get("p")
            if p_val is None:
                node["p"] = 0.0
            elif isinstance(p_val, (int, float)):
                node["p"] = float(p_val)
            elif isinstance(p_val, list):
                # ベクトルが与えられた場合はそのまま（長さチェックは行わない）
                node["p"] = p_val
        elif "p" in node:
            del node["p"]

        # t の既定値・形式（どちらかに関与するノード）
        if node["key"] in thermal_nodes or node["key"] in ventilation_nodes:
            t_val = node.get("t")
            if t_val is None:
                node["t"] = 20.0
            elif isinstance(t_val, (int, float)):
                node["t"] = float(t_val)
            elif isinstance(t_val, list):
                node["t"] = t_val
        elif "t" in node:
            del node["t"]

        # エアコン特有の pre_temp
        if node.get("type") == NodeTypeEnum.AIRCON:
            pre = node.get("pre_temp")
            if pre is None:
                node["pre_temp"] = 20.0
            elif isinstance(pre, (int, float)):
                node["pre_temp"] = float(pre)
            elif isinstance(pre, list):
                node["pre_temp"] = pre
        elif "pre_temp" in node:
            del node["pre_temp"]

        # 参照ノード
        if "ref_node" in node:
            result = validate_node_exists(node["ref_node"], node_config, f"ノード {node['key']} の参照先")
            if not result.is_valid:
                errors.extend(result.errors)

        # 体積・沈着率（必要時）
        if sim_config["calc_flag"]["x"] or sim_config["calc_flag"]["c"]:
            node["v"] = node.get("v", 0.0)
        if sim_config["calc_flag"]["c"]:
            node["beta"] = node.get("beta", 0.0)

    logger.info("node_configのバリデーションが完了しました。")
    return node_config, ValidationResult(len(errors) == 0, errors, warnings)


def validate_ventilation_config(
    sim_config: SimConfigType,
    node_config: List[NodeType],
    ventilation_config: List[Dict[str, Any]],
) -> Tuple[List[VentilationBranchType], ValidationResult]:
    logger.info("ventilation_configのバリデーションを開始します。")

    if not isinstance(ventilation_config, (list, np.ndarray)):
        return ventilation_config, ValidationResult(False, ["ventilation_configはリストまたは配列である必要があります"], [])
    if isinstance(ventilation_config, np.ndarray):
        ventilation_config = ventilation_config.tolist()

    errors: List[str] = []
    warnings: List[str] = []

    branch_types = {
        VentilationBranchTypeEnum.SIMPLE_OPENING: {"required": ["alpha", "area"]},
        VentilationBranchTypeEnum.GAP: {"required": ["a", "n"]},
        VentilationBranchTypeEnum.FAN: {"required": ["p_max", "q_max", "p1", "q1"]},
        VentilationBranchTypeEnum.FIXED_FLOW: {"required": ["vol"]},
    }

    for branch in ventilation_config:
        result = validate_required_fields(branch, ["key"], "換気ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # 未知キーは事前に除去（タイプ判定に影響しないものは捨てる）
        log_and_strip_unknown_fields(branch, _allowed_keys(VentilationBranchType), f"換気ブランチ {branch.get('key', '?')}")

        source, target = branch["key"].split(CHAIN_DELIMITER)
        branch["source"], branch["target"] = source, target

        result = validate_node_chain(branch["key"], source, target, "換気ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        result = validate_node_exists(source, node_config, f"換気ブランチ {branch['key']} の'source'")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        result = validate_node_exists(target, node_config, f"換気ブランチ {branch['key']} の'target'")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # enable はデフォルトでスカラー True（巨大化回避）
        branch["enable"] = branch.get("enable", True)

        # タイプ判定と必須確認
        result = validate_branch_type(branch, branch_types, "換気ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        result = validate_branch_parameters(branch, branch_types, "換気ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # FIXED_FLOW の vol はスカラーのまま（配列が来た場合はそのまま）
        if branch["type"] == VentilationBranchTypeEnum.FIXED_FLOW:
            vol_value = branch.get("vol")
            if isinstance(vol_value, (int, float)):
                branch["vol"] = float(vol_value)
            elif isinstance(vol_value, list):
                # ベクトルが来た場合は長さチェックを行わず受け入れ
                branch["vol"] = vol_value
            else:
                errors.append(f"換気ブランチ {branch['key']} の'vol'は数値または配列である必要があります")

        # 付加情報
        branch["h_from"] = branch.get("h_from", 0.0)
        branch["h_to"] = branch.get("h_to", 0.0)
        # set_default_generation(branch, sim_config, "eta", "c")  # 必要に応じて

    logger.info("ventilation_configのバリデーションが完了しました。")
    return ventilation_config, ValidationResult(len(errors) == 0, errors, warnings)


def validate_thermal_config(
    sim_config: SimConfigType,
    node_config: List[NodeType],
    thermal_config: List[Dict[str, Any]],
) -> Tuple[List[ThermalBranchType], ValidationResult]:
    logger.info("thermal_configのバリデーションを開始します。")

    if not isinstance(thermal_config, (list, np.ndarray)):
        return thermal_config, ValidationResult(False, ["thermal_configはリストまたは配列である必要があります"], [])
    if isinstance(thermal_config, np.ndarray):
        thermal_config = thermal_config.tolist()

    errors: List[str] = []
    warnings: List[str] = []

    branch_types = {
        ThermalBranchTypeEnum.CONDUCTANCE: {"required": ["conductance"], "optional": ["u_value", "area"]},
        ThermalBranchTypeEnum.HEAT_GENERATION: {"required": ["heat_generation"]},
    }

    for branch in thermal_config:
        result = validate_required_fields(branch, ["key"], "熱ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # 未知キーを削除
        log_and_strip_unknown_fields(branch, _allowed_keys(ThermalBranchType), f"熱ブランチ {branch.get('key', '?')}")

        source, target = branch["key"].split(CHAIN_DELIMITER)
        if source == "":
            source = "void"
        branch["source"], branch["target"] = source, target

        result = validate_node_chain(branch["key"], source, target, "熱ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        if source != "void":
            result = validate_node_exists(source, node_config, f"熱ブランチ {branch['key']} の'source'")
            if not result.is_valid:
                errors.extend(result.errors)
                continue

        result = validate_node_exists(target, node_config, f"熱ブランチ {branch['key']} の'target'")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        # enable はデフォルトでスカラー True
        branch["enable"] = branch.get("enable", True)

        # 自動タイプ判定（u_value × area → conductance）
        if "type" not in branch:
            if "conductance" in branch:
                branch["type"] = ThermalBranchTypeEnum.CONDUCTANCE
            elif "u_value" in branch and "area" in branch:
                branch["type"] = ThermalBranchTypeEnum.CONDUCTANCE
                branch["conductance"] = float(branch["u_value"]) * float(branch["area"])
            elif "heat_generation" in branch:
                branch["type"] = ThermalBranchTypeEnum.HEAT_GENERATION
            else:
                errors.append(
                    f"熱ブランチ {branch['key']} に'conductance'または'heat_generation'が指定されていません"
                )
                continue

        result = validate_branch_type(branch, branch_types, "熱ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

        result = validate_branch_parameters(branch, branch_types, "熱ブランチ")
        if not result.is_valid:
            errors.extend(result.errors)
            continue

    logger.info("thermal_configのバリデーションが完了しました。")
    return thermal_config, ValidationResult(len(errors) == 0, errors, warnings)


# ------------------------------
# エントリポイント
# ------------------------------
def validate(config_path: str, *, continue_on_error: bool = True) -> Dict[str, Any]:
    """
    指定した JSON 設定のバリデーションを行い、正規化後の JSON を返す。
    成功時は temp ディレクトリに JSON を保存する。
    """
    logger.info("validationを開始します。")

    # 設定ファイルの基本検証
    raw_config = validate_config_file(config_path)

    sim_config = raw_config["simulation"]
    node_config = raw_config["nodes"]
    ventilation_config = raw_config["ventilation_branches"]
    thermal_config = raw_config["thermal_branches"]

    # セクション別バリデーション（集約モード対応）
    all_errors: List[str] = []

    sim_config, sim_result = validate_sim_config(sim_config)
    if not sim_result.is_valid:
        all_errors.extend(sim_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    node_config, node_result = validate_node_config(sim_config, node_config, ventilation_config, thermal_config)
    if not node_result.is_valid:
        all_errors.extend(node_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    ventilation_config, vent_result = validate_ventilation_config(sim_config, node_config, ventilation_config)
    if not vent_result.is_valid:
        all_errors.extend(vent_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    thermal_config, thermal_result = validate_thermal_config(sim_config, node_config, thermal_config)
    if not thermal_result.is_valid:
        all_errors.extend(thermal_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    if all_errors:
        raise ValidationError("\n".join(all_errors))

    # 出力組み立て
    output_json: Dict[str, Any] = {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
    }
    output_json = convert_to_json_compatible(output_json)

    # temp へ保存
    input_stem = Path(config_path).stem.replace("_02", "")
    output_dir = Path(__file__).resolve().parent.parent / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    logger.info("validationが正常に終了しました。")
    return output_json


def validate_dict(config: Dict[str, Any], *, continue_on_error: bool = True) -> Dict[str, Any]:
    """
    既にロード済みの設定 dict を検証・正規化し、JSON 互換化した dict を返す。
    ファイル I/O は行わない。
    """
    # 必須セクション確認
    for section in ["simulation", "nodes", "ventilation_branches", "thermal_branches"]:
        if section not in config:
            raise ConfigFileError(f"設定データに'{section}'セクションがありません")

    sim_config = config["simulation"]
    node_config = config["nodes"]
    ventilation_config = config["ventilation_branches"]
    thermal_config = config["thermal_branches"]

    all_errors: List[str] = []

    sim_config, sim_result = validate_sim_config(sim_config)
    if not sim_result.is_valid:
        all_errors.extend(sim_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    node_config, node_result = validate_node_config(sim_config, node_config, ventilation_config, thermal_config)
    if not node_result.is_valid:
        all_errors.extend(node_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    ventilation_config, vent_result = validate_ventilation_config(sim_config, node_config, ventilation_config)
    if not vent_result.is_valid:
        all_errors.extend(vent_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    thermal_config, thermal_result = validate_thermal_config(sim_config, node_config, thermal_config)
    if not thermal_result.is_valid:
        all_errors.extend(thermal_result.errors)
        if not continue_on_error:
            raise ValidationError("\n".join(all_errors))

    if all_errors:
        raise ValidationError("\n".join(all_errors))

    output_json: Dict[str, Any] = {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
    }
    return convert_to_json_compatible(output_json)



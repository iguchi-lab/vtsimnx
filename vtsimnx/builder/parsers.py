from __future__ import annotations

from typing import Any, Dict, List
import json
import os

from .logger import get_logger
from .config_types import SimConfigType
from .utils import (
    _split_key_and_comment,
    _split_compound_key,
    _expand_chain,
    _append_with_comment,
)

logger = get_logger(__name__)


def _parse_simulation(raw: Dict[str, Any]) -> SimConfigType:
    logger.info("シミュレーション設定の解析を開始します")
    sim: SimConfigType = {
        "index": {"start": "", "end": "", "timestep": 0, "length": 0},
        "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        "calc_flag": {"p": False, "t": False, "x": False, "c": False},
    }
    sim_data = raw.get("simulation", {})
    if not isinstance(sim_data, dict):
        raise TypeError("simulation セクションは dict である必要があります。")

    if "index" in sim_data and isinstance(sim_data["index"], dict):
        sim["index"].update(sim_data["index"])
    if "tolerance" in sim_data and isinstance(sim_data["tolerance"], dict):
        sim["tolerance"].update(sim_data["tolerance"])
    return sim


def _parse_nodes(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("ノードの解析を開始します")
    node_config: List[Dict[str, Any]] = [{"key": "void"}]

    nodes = raw.get("nodes")
    if nodes is None:
        raise ValueError("ノードの設定が見つかりませんでした。")
    if not isinstance(nodes, list):
        raise TypeError("nodes セクションは list である必要があります。")

    for node in nodes:
        if not isinstance(node, dict) or "key" not in node:
            raise ValueError(f"不正なノード定義: {node!r}")
        key_raw = str(node["key"])
        key_no_comment, comment = _split_key_and_comment(key_raw)
        for sub_key in _split_compound_key(key_no_comment):
            if sub_key == "void":
                raise ValueError("ノード名には 'void' を使用できません。")
            node_dict = _append_with_comment(node, key=sub_key, comment=comment)
            node_config.append(node_dict)
            logger.info(f"ノードを解析しました: {sub_key}")

    return node_config


def _parse_chain_branches(raw: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    items = raw.get(field)
    config: List[Dict[str, Any]] = []
    if items is None:
        logger.info(f"{field} の設定が見つかりませんでした。")
        return config

    if not isinstance(items, list):
        raise TypeError(f"{field} セクションは list である必要があります。")

    logger.info(f"{field} の解析を開始します")
    for branch in items:
        if not isinstance(branch, dict) or "key" not in branch:
            raise ValueError(f"不正なブランチ定義（{field}）: {branch!r}")
        key_raw = str(branch["key"])
        key_no_comment, comment = _split_key_and_comment(key_raw)
        for sub_key in _expand_chain(key_no_comment):
            branch_dict = _append_with_comment(branch, key=sub_key, comment=comment)
            config.append(branch_dict)
            logger.info(f"{field} を解析しました: {sub_key}")
    return config


def _parse_surface(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = raw.get("surfaces")
    surface_config: List[Dict[str, Any]] = []
    if items is None:
        logger.info("表面の設定が見つかりませんでした。")
        return surface_config

    if not isinstance(items, list):
        raise TypeError("surface セクションは list である必要があります。")

    logger.info("表面の解析を開始します")
    for surface in items:
        if not isinstance(surface, dict) or "key" not in surface:
            raise ValueError(f"不正な表面定義: {surface!r}")
        key_raw = str(surface["key"])
        key_no_comment, comment = _split_key_and_comment(key_raw)
        surface_config.append(_append_with_comment(surface, key=key_no_comment, comment=comment))
    return surface_config


def _parse_aircon(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = raw.get("aircon")
    if items is None:
        logger.info("エアコンの設定が見つかりませんでした。")
        return []
    if not isinstance(items, list):
        raise TypeError("aircon セクションは list である必要があります。")

    from .utils import _normalize_timeseries_mapping

    logger.info("エアコンの解析を開始します")
    out: List[Dict[str, Any]] = []
    for ac in items:
        if not isinstance(ac, dict):
            raise ValueError(f"不正なエアコン定義: {ac!r}")
        out.append(_normalize_timeseries_mapping(ac))
    return out


def parse_all(raw: Dict[str, Any]) -> tuple[SimConfigType, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    各セクションのパースを一括で行い、タプルで返す。
    戻り値: (sim_config, node_config, ventilation_config, thermal_config, surface_config, aircon_config)
    """
    # 入力が文字列（JSON文字列またはファイルパス）の場合に辞書へ変換
    if isinstance(raw, (bytes, bytearray)):
        logger.info("raw が bytes のため UTF-8 デコードして JSON パースを試みます")
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        logger.info("raw が文字列のため JSON パース（失敗時はファイル読み込み）を試みます")
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            if os.path.exists(raw):
                with open(raw, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            else:
                raise TypeError("raw は dict、JSON文字列、または既存のJSONファイルパスである必要があります。")
    if not isinstance(raw, dict):
        raise TypeError("raw は dict である必要があります。")

    sim_config = _parse_simulation(raw)
    node_config = _parse_nodes(raw)
    ventilation_config = _parse_chain_branches(raw, "ventilation_branches")
    thermal_config = _parse_chain_branches(raw, "thermal_branches")
    surface_config = _parse_surface(raw)
    aircon_config = _parse_aircon(raw)
    return sim_config, node_config, ventilation_config, thermal_config, surface_config, aircon_config



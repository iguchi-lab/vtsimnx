from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import json
import numpy as np
import pandas as pd

from .logger import get_logger
from .config_types import SimConfigType, IndexType, ToleranceType, CalcFlagType

logger = get_logger(__name__)

# ------------------------------
# 定数（区切り文字）
# ------------------------------
CHAIN_DELIMITER = "->"      # ノード連鎖の区切り
COMMENT_DELIMITER = "||"    # インラインコメントの区切り
COMPOUND_DELIMITER = "&&"   # 複合キー（AND条件）の区切り


# ------------------------------
# ユーティリティ
# ------------------------------
def _split_key_and_comment(key: str) -> Tuple[str, str]:
    """
    'A||comment' → ('A', 'comment')
    コメントがなければ ('A', '') を返す。
    """
    k = key.strip()
    if COMMENT_DELIMITER in k:
        head, tail = k.split(COMMENT_DELIMITER, 1)
        return head.strip(), tail.strip()
    return k, ""


def _split_compound_key(key: str, delimiter: str = COMPOUND_DELIMITER) -> List[str]:
    """
    'A&&B' → ['A', 'B']。デリミタがなければ [key]。
    """
    k = key.strip()
    if delimiter in k:
        return [part.strip() for part in k.split(delimiter)]
    return [k]


def _expand_chain(key: str) -> List[str]:
    """
    'A->B->C' → ['A->B', 'B->C']
    空文字 '' は 'void' とみなす（'->B' 等の表記許容）。
    """
    nodes = [n.strip() for n in key.split(CHAIN_DELIMITER)]
    if len(nodes) < 2:
        raise ValueError(f"連鎖の定義が短すぎます: '{key}'")
    segs: List[str] = []
    for i in range(len(nodes) - 1):
        left = nodes[i] if nodes[i] else "void"
        right = nodes[i + 1] if nodes[i + 1] else "void"
        segs.append(f"{left}{CHAIN_DELIMITER}{right}")
    return segs


def _normalize_timeseries_mapping(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    dict の値が pandas.Series / numpy.ndarray なら list に変換。
    dict は非破壊（シャローコピー）で返す。
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, pd.Series):
            out[k] = v.tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _append_with_comment(base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """
    base に overrides をマージ（上書き）。Series/ndarray は list 化。
    """
    merged = {**base, **overrides}
    return _normalize_timeseries_mapping(merged)


# ------------------------------
# セクション別パーサ
# ------------------------------
def _parse_simulation(raw: Dict[str, Any]) -> SimConfigType:
    """simulation セクションに既定値を適用。"""
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
    # calc_flag は後段で nodes から自動算定（ここでは初期値のまま）
    return sim


def _parse_nodes(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """nodes セクションの展開。'void' ノードは予約語として自動追加のみ許可。"""
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
    """
    ventilation_branches / thermal_branches で共通の分解処理。
    """
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
    """surface セクションの解析。キーとコメントを分離して付加。"""
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
    """aircon セクション（定義があれば list を返す。なければ空配列）。"""
    items = raw.get("aircon")
    if items is None:
        logger.info("エアコンの設定が見つかりませんでした。")
        return []
    if not isinstance(items, list):
        raise TypeError("aircon セクションは list である必要があります。")

    logger.info("エアコンの解析を開始します")
    out: List[Dict[str, Any]] = []
    for ac in items:
        if not isinstance(ac, dict):
            raise ValueError(f"不正なエアコン定義: {ac!r}")
        # 特にキー展開はない前提。Series/ndarray の list 化のみ行う
        out.append(_normalize_timeseries_mapping(ac))
    return out


# ------------------------------
# エントリポイント
# ------------------------------
def parse(raw_config: Dict[str, Any], output_path: Optional[str] = "parsed_input_data.json") -> Dict[str, Any]:
    """
    設定 raw_config を正規化して dict を返す。
    output_path を None にするとファイル出力しない。
    """
    logger.info("設定データの読み込み開始")
    try:
        # ※ raw_config は外部で再利用される可能性もあるため非破壊の方針
        raw = deepcopy(raw_config)

        output_json: Dict[str, Any] = {
            "simulation":           _parse_simulation(raw),
            "nodes":                _parse_nodes(raw),
            "ventilation_branches": _parse_chain_branches(raw, "ventilation_branches"),
            "thermal_branches":     _parse_chain_branches(raw, "thermal_branches"),
            "surfaces":             _parse_surface(raw),
            "aircon":               _parse_aircon(raw),
        }

        # 計算フラグの自動設定（nodes の calc_p/calc_t/calc_x/calc_c を集計）
        for flag in ("p", "t", "x", "c"):
            has_flag = any(
                isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
                for node in output_json["nodes"]
            )
            output_json["simulation"]["calc_flag"][flag] = has_flag

        logger.info("設定データの処理が完了しました")

        # JSON 出力（必要に応じて）
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)
            logger.info(f"設定データを {output_path} に出力しました")

        return output_json

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        # そのまま上げる：上位で適切にハンドリングさせる
        raise

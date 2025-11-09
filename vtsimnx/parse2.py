from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from .config_types import SimConfigType, IndexType, ToleranceType, CalcFlagType
from .logger import get_logger

logger = get_logger(__name__)

# ------------------------------
# 文字列区切り
# ------------------------------
CHAIN_DELIMITER = "->"     # ノード連鎖の区切り
COMMENT_DELIMITER = "||"   # インラインコメントの区切り
COMPOUND_DELIMITER = "&&"  # 複合キー（AND条件）の区切り


# ========= ヘルパ =========

def _split_key_and_comment(key: str) -> Tuple[str, str]:
    """キー文字列からコメント区切りを取り除き、(key, comment) を返す。"""
    if COMMENT_DELIMITER in key:
        k, c = key.split(COMMENT_DELIMITER, 1)
        return k.strip(), c.strip()
    return key.strip(), ""

def _expand_chain(key: str) -> List[str]:
    """'A->B->C' を ['A->B', 'B->C'] に分解。"""
    nodes = [n.strip() for n in key.split(CHAIN_DELIMITER)]
    if len(nodes) < 2:
        raise ValueError(f"連鎖の定義が短すぎます: '{key}'")
    return [f"{nodes[i]}{CHAIN_DELIMITER}{nodes[i+1]}" for i in range(len(nodes) - 1)]

def _ensure_no_nested_lists(items: List[Any], label: str) -> None:
    """ブランチ配列の入れ子構造を禁止。"""
    for i, it in enumerate(items):
        if isinstance(it, list):
            raise ValueError(f"{label}[{i}] に入れ子のリスト構造が検出されました。フラットなリストにしてください。")

def _normalize_timeseries(value: Any, df: Optional[Any], logtag: str) -> Any:
    """
    時系列値の正規化:
      - str かつ df があり df.columns にあれば df[col].to_list()
      - pandas.Series なら .to_list()
      - numpy 配列等なら .tolist()
      - list/スカラーはそのまま返す
    """
    # 1) "列名" 指定
    if isinstance(value, str) and df is not None:
        cols = getattr(df, "columns", [])
        if value in cols:
            logger.info(f"データ列を読み込みました: {value}（{logtag}）")
            return df[value].to_list()
        logger.warning(f"指定列 '{value}' は df に見つかりません（{logtag}）。文字列のまま扱います。")
        return value

    # 2) pandas.Series 想定（duck-typing）
    if hasattr(value, "to_list"):
        try:
            return value.to_list()
        except Exception:
            pass  # 失敗時は次へ

    # 3) numpy 配列など
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    # 4) list/スカラー等はそのまま
    return value

def _compile_sim_config(raw: Dict[str, Any]) -> SimConfigType:
    """simulation セクションの既定値を用意し、raw の値で update。"""
    sim: SimConfigType = {
        "index": {"start": "", "end": "", "timestep": 0, "length": 0},
        "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        "calc_flag": {"p": False, "t": False, "x": False, "c": False},
    }
    sim_data = raw.get("simulation", {})
    if "index" in sim_data:
        sim["index"].update(sim_data["index"])
    if "tolerance" in sim_data:
        sim["tolerance"].update(sim_data["tolerance"])
    return sim

def _parse_nodes(raw: Dict[str, Any], df: Optional[Any]) -> List[Dict[str, Any]]:
    """nodes セクションの解析（複合キー展開、コメント抽出、p/t の正規化など）。"""
    if "nodes" not in raw:
        raise ValueError("ノードの設定が見つかりませんでした。")

    logger.info("ノードの解析を開始します")
    node_config: List[Dict[str, Any]] = [{"key": "void"}]  # デフォルトノード

    for node in raw["nodes"]:
        resolved = dict(node)

        # p/t の正規化（Series/ndarray→list、列名→df[col].to_list()）
        for k in ("p", "t"):
            if k in resolved:
                resolved[k] = _normalize_timeseries(resolved[k], df, f"nodes[{node.get('key','?')}].{k}")

        key_raw = resolved["key"]
        key_body, comment = _split_key_and_comment(key_raw)

        # 複合キー 'A&&B' → ['A', 'B']
        parts = [s.strip() for s in key_body.split(COMPOUND_DELIMITER)] if COMPOUND_DELIMITER in key_body else [key_body]

        for sub_key in parts:
            if sub_key == "void":
                raise ValueError("ノード名には 'void' を使用できません。")
            node_config.append({**resolved, "key": sub_key, "comment": comment})
            logger.info(f"ノードを解析しました: {sub_key}")

    return node_config

def _parse_branches(
    raw_list: List[Dict[str, Any]],
    *,
    label: str,
    allow_void_on_empty_head: bool = False
) -> List[Dict[str, Any]]:
    """
    換気/熱ブランチの共通解析。
    - キーからコメント抽出
    - チェーン展開
    - 熱ブランチのみ、先頭ノードが空文字なら 'void' に置換可能
    """
    if not raw_list:
        logger.info(f"{label} の設定が見つかりませんでした。")
        return []

    logger.info(f"{label} の解析を開始します")
    _ensure_no_nested_lists(raw_list, label)
    out: List[Dict[str, Any]] = []

    for branch in raw_list:
        k_raw = branch["key"]
        k_body, comment = _split_key_and_comment(k_raw)

        # 熱ブランチでのみ、先頭空文字→'void' 置換
        if allow_void_on_empty_head and k_body.startswith(CHAIN_DELIMITER):
            k_body = f"void{k_body}"

        try:
            segments = _expand_chain(k_body)
        except ValueError as e:
            raise ValueError(f"{label} の key が不正です: '{k_raw}': {e}") from e

        for seg in segments:
            out.append({**branch, "key": seg, "comment": comment})
            logger.info(f"{label} を解析しました: {seg}")

    return out

def _parse_surfaces(raw: Dict[str, Any], df: Optional[Any]) -> List[Dict[str, Any]]:
    """surfaces セクションの解析（コメント抽出、solar の正規化）。"""
    surfaces = raw.get("surfaces", [])
    if not surfaces:
        logger.info("表面の設定が見つかりませんでした。")
        return []

    logger.info("表面の解析を開始します")
    out: List[Dict[str, Any]] = []

    for surface in surfaces:
        resolved = dict(surface)
        if "solar" in resolved:
            resolved["solar"] = _normalize_timeseries(resolved["solar"], df, f"surfaces[{surface.get('key','?')}].solar")
        key_raw = resolved["key"]
        key_body, comment = _split_key_and_comment(key_raw)
        out.append({**resolved, "key": key_body, "comment": comment})

    return out

def _set_calc_flags(sim: SimConfigType, nodes: List[Dict[str, Any]]) -> None:
    """calc_flag をノードの calc_* から自動設定。"""
    for flag in ("p", "t", "x", "c"):
        sim["calc_flag"][flag] = any(n.get(f"calc_{flag}", False) for n in nodes)


# ========= メイン =========

def parse2(raw_config: Dict[str, Any], df: Optional[Any] = None) -> Dict[str, Any]:
    """
    設定データの読み込み・整形。
    - raw_config は dict 前提で、破壊的変更をしない
    - p/t（nodes）や solar（surfaces）が:
        * pandas.Series → list に変換
        * numpy.ndarray → list に変換
        * "列名"（str）かつ df 提供 → df[col].to_list()
        * list/スカラー → そのまま
    """
    logger.info("設定データの読み込み開始")
    try:
        # 破壊的変更を避ける
        raw = deepcopy(raw_config)

        # simulation
        sim_config = _compile_sim_config(raw)

        # nodes（p/t 正規化を含む）
        node_config = _parse_nodes(raw, df)

        # branches
        ventilation_config = _parse_branches(
            raw.get("ventilation_branches", []),
            label="換気ブランチ",
        )
        thermal_config = _parse_branches(
            raw.get("thermal_branches", []),
            label="熱ブランチ",
            allow_void_on_empty_head=True,  # 先頭空文字→void
        )

        # surfaces（solar 正規化を含む）
        surface_config = _parse_surfaces(raw, df)

        # aircon（そのまま）
        aircon_config = raw.get("aircon", [])
        if aircon_config:
            logger.info("エアコンの解析を開始します")
        else:
            logger.info("エアコンの設定が見つかりませんでした。")

        # calc_flag の自動設定
        _set_calc_flags(sim_config, node_config)

        output = {
            "simulation": sim_config,
            "nodes": node_config,
            "ventilation_branches": ventilation_config,
            "thermal_branches": thermal_config,
            "surfaces": surface_config,
            "aircon": aircon_config,
        }

        logger.info("設定データの処理が完了しました")
        return output

    except Exception as e:
        logger.exception("エラーが発生しました: %s", e)
        raise

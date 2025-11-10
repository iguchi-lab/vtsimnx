from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import json
import re
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
# 表面の種類の対応関係
# ------------------------------
SURFACE_PAIR = {
    "wall": "wall",
    "floor": "ceiling",
    "ceiling": "floor",
    "glass": "glass",
}

DEFAULT_ALPHA_I = 4.4  # 室内側表面の対流熱伝達率
DEFAULT_ALPHA_O = 17.9  # 室外側表面の対流熱伝達率
DEFAULT_ALPHA_R = 4.6  # 放射熱伝達率

_INT_RE   = re.compile(r'^[+-]?\d+\Z')
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')

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

def convert_numeric_values(
    obj: Any,
    *,
    bool_keys: Optional[Iterable[str]] = None,
    _parent_key: Optional[str] = None,   # 使わないなら削ってOK
) -> Any:
    """
    JSON由来のオブジェクトを走査して、数値っぽい値を適切な型へ変換する。
      - dict : 再帰。bool化したいキー名は bool_keys に指定（例: {'calc_p', ...}）
      - list/tuple: 全要素が数値なら np.array に（空シーケンスはそのまま）
      - str  : 整数/浮動小数表現なら数値化（科学記法含む）
      - int/float/bool/np.number: そのまま
    """
    bool_keys_set = set(bool_keys or ())

    # dict
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            converted = convert_numeric_values(v, bool_keys=bool_keys_set, _parent_key=str(k))
            # 0/1 を bool にしたいキーだけ変換（NumPy整数も考慮）
            if (
                str(k) in bool_keys_set
                and isinstance(converted, (int, np.integer))
                and converted in (0, 1)
            ):
                out[k] = bool(converted)
            else:
                out[k] = converted
        return out

    # list/tuple → 再帰してから一括判定
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return obj  # 空はAPI都合でそのまま
        converted = [convert_numeric_values(x, bool_keys=bool_keys_set, _parent_key=_parent_key) for x in obj]
        # すべて数値なら np.array 化（np.bool_ は除外）
        if all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
               for x in converted):
            return np.array(converted)
        return converted if isinstance(obj, list) else tuple(converted)

    # 文字列 -> 数値っぽければ変換
    if isinstance(obj, str):
        s = obj.strip()
        if _INT_RE.match(s):
            return int(s)
        if _FLOAT_RE.match(s):
            return float(s)
        return obj

    # すでに bool はそのまま（bool は int のサブクラスなので先に判定）
    if isinstance(obj, (bool, np.bool_)):
        return obj

    # 数値スカラーはそのまま
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return obj

    # それ以外は無変換
    return obj

def convert_to_json_compatible(obj: Any) -> Any:
    """
    Recursively converts numpy objects within a dictionary or list
    to their standard Python equivalents for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_compatible(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    else:
        return obj

def ensure_timeseries(value, length: int):
    """スカラーならタイムステップ長の配列に展開し、配列/ndarrayはそのまま返す"""
    if isinstance(value, (list, np.ndarray)):
        return list(value) # Ensure list output
    return [value] * length

def get_node_prefix(surface: dict) -> tuple[str, str]:
    """ノードの接頭辞を取得する"""
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
    """表面を処理する"""
    nodes = []
    thermal_branches = []

    # ブランチの両端のノードを取得
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

    # ノードの追加
    for i, node in enumerate(node_names):
        logger.info(f"　ノード【{node}】 を追加します。")
        nodes.append(
            {"key": node, "calc_t": True, "thermal_mass": thermal_mass[i], "subtype": node_types[i]}
        )

    for i, branch in enumerate(thermal_branch_names):
        logger.info(f"　熱ブランチ【{branch}】を追加します。")
        thermal_branches.append(
            {"key": branch, "conductance": conductance[i], "subtype": branch_types[i]}
        )

    return nodes, thermal_branches

def process_wall_solar(surface: dict, sim_length: int) -> list:
    """壁面の日射を処理する"""
    thermal_branches = []

    # ブランチの両端のノードを取得
    _, _, _, o_prefix = get_node_prefix(surface)

    # Use numpy for element-wise multiplication
    heat_generation = surface["area"] * surface.get("eta", 1.0) * np.array(surface["solar"])
    heat_generation = ensure_timeseries(heat_generation, sim_length)

    # 日射熱ブランチの追加
    branch_key = f"void->{o_prefix}_s"
    logger.info(f"　外壁日射熱ブランチ【{branch_key}】を追加します。")
    thermal_branches.append(
        {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain"}
    )
    return thermal_branches


def process_glass_solar(surface: dict, surfaces: list, sim_length: int) -> list:
    """日射を処理する"""
    thermal_branches = []

    # 日射熱ブランチの追加
    node = surface["key"].split(CHAIN_DELIMITER)[0]

    # ノードのキーで始まる表面を検索
    surfaces_of_target_node = [s for s in surfaces if s["key"].startswith(node)]
    area_ceiling = sum(
        [s["area"] for s in surfaces_of_target_node if s["part"] == "ceiling"]
    )
    area_wall = sum(
        [s["area"] for s in surfaces_of_target_node if s["part"] == "wall"]
    )
    area_ceiling_wall = area_ceiling + area_wall
    area_floor = sum(
        [s["area"] for s in surfaces_of_target_node if s["part"] == "floor"]
    )

    # Use numpy for element-wise multiplication
    heat_generation_floor        = np.array(surface["solar"]) * surface["area"] * 0.50 * surface["eta"]
    heat_generation_ceiling_wall = np.array(surface["solar"]) * surface["area"] * 0.50 * surface["eta"]

    heat_generation_floor        = ensure_timeseries(heat_generation_floor,        sim_length)
    heat_generation_ceiling_wall = ensure_timeseries(heat_generation_ceiling_wall, sim_length)

    # 日射熱ブランチの追加
    for s in surfaces_of_target_node:
        # ブランチの両端のノードを取得
        _, _, i_prefix, _ = get_node_prefix(s)
        branch_key = f"void->{i_prefix}_s"
        if s["part"] == "floor":
            # Ensure the result is a list before appending
            heat_generation = (np.array(heat_generation_floor) * s["area"] / area_floor).tolist()
        elif s["part"] == "ceiling" or s["part"] == "wall":
            # Ensure the result is a list before appending
            heat_generation = (np.array(heat_generation_ceiling_wall) * s["area"] / area_ceiling_wall).tolist()
        else:
            continue
        logger.info(f"　ガラス透過日射熱ブランチ【{branch_key}】を追加します。")
        thermal_branches.append(
            {"key": branch_key, "heat_generation": heat_generation, "subtype": "solar_gain",}
        )

    return thermal_branches

def process_radiation(node: str, surfaces: list) -> list:
    """放射を処理する"""
    thermal_branches = []

    # 表面ノードのキーのリストを取得
    surface_nodes = [f"{get_node_prefix(s)[2]}_s" for s in surfaces]
    sum_area = sum([s["area"] for s in surfaces])

    # すべての組み合わせを生成
    for i, node1 in enumerate(surface_nodes):
        for j, node2 in enumerate(
            surface_nodes[i + 1 :], start=i + 1
        ):  # i+1以降のノードとの組み合わせ
            branch_key = f"{node1}->{node2}"
            area1 = surfaces[i]["area"]
            area2 = surfaces[j]["area"]
            conductance = DEFAULT_ALPHA_R * area1 * area2 / sum_area
            logger.info(f"　室内放射熱ブランチ【{branch_key}】を追加します。")
            thermal_branches.append(
                {"key": branch_key, "conductance": conductance, "subtype": "radiation"}
            )

    return thermal_branches

def process_aircon(aircon: dict) -> tuple[list, list]:
    """空調を処理する"""
    nodes = []
    ventilation_branches = []

    in_node = aircon.get("in", aircon["set"])
    aircon_out_node = f"{aircon['key']}"
    out_node = aircon.get("out", aircon["set"])
    set_node = aircon["set"]
    outside_node = aircon["outside"]
    pre_temp = aircon["pre_temp"]
    model = aircon["model"]
    mode = aircon["mode"]

    ventilation_chain = [
        f"{in_node}->{aircon_out_node}",
        f"{aircon_out_node}->{out_node}",
    ]
    vol = aircon.get("vol", 1000 / 3600)

    # ノードの追加
    logger.info(f"　エアコンノード【{aircon_out_node}】を追加します。")
    nodes.append(
        {
            "key": aircon_out_node,
            "calc_t": True,
            "in_node": in_node,
            "set_node": set_node,
            "outside_node": outside_node,
            "type": "aircon",
            "pre_temp": pre_temp,
            "model": model,
            "mode": mode,
            "ac_spec": aircon.get("ac_spec", {}),
        }
    )

    for branch in ventilation_chain:
        logger.info(f"　換気ブランチ【{branch}】を追加します。")
        ventilation_branches.append({"key": branch, "vol": vol, "subtype": "aircon"})

    return nodes, ventilation_branches

def add_capacity(    node: dict, time_step: float) -> tuple[list, list]:
    """熱容量を追加する"""
    nodes = []
    thermal_branches = []

    logger.info(f"　熱容量ノード【{node['key']}_c】を追加します。")
    nodes.append(
        {"key": f"{node['key']}_c",
         "calc_t": False, "type": "capacity", "ref_node": node["key"]}
    )

    logger.info(f"　熱容量ブランチ【{node['key']}_c->{node['key']}】を追加します。")
    thermal_branches.append(
        {"key": f"{node['key']}_c->{node['key']}",
         "conductance": node["thermal_mass"] / time_step, "subtype": "capacity"}
    )
    return nodes, thermal_branches


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

        sim_config =  _parse_simulation(raw)
        node_config = _parse_nodes(raw)
        ventilation_config = _parse_chain_branches(raw, "ventilation_branches")
        thermal_config = _parse_chain_branches(raw, "thermal_branches")
        surface_config = _parse_surface(raw)
        aircon_config = _parse_aircon(raw)

        # 計算フラグの自動設定（nodes の calc_p/calc_t/calc_x/calc_c を集計）
        logger.info("計算フラグの自動設定を開始します")
        for flag in ("p", "t", "x", "c"):
            has_flag = any(
                isinstance(node, dict) and bool(node.get(f"calc_{flag}", False))
                for node in node_config
            )
            sim_config["calc_flag"][flag] = has_flag

        logger.info("設定データの処理が完了しました")

        if "surfaces" in raw_config:
            logger.info("表面の解析を開始します。")
            surface_data = raw_config["surfaces"]
            for surface in surface_data:
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

            # 設定ファイルの保存
            output_json = {
                "simulation": sim_config,
                "nodes": node_config,
                "ventilation_branches": ventilation_config,
                "thermal_branches": thermal_config,
            }

            # Convert to JSON-compatible format before dumping
            output_json = convert_to_json_compatible(output_json)

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

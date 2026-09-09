from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)

_FAN_KEYS = ("p_max", "p1", "q1", "q_max")
# PQ 時の空調→吹出は短いつなぎ。室への分け前は利用者が書く pressure_loss に任せる。
_DEFAULT_CONNECTOR_AREA = 0.05
_DEFAULT_CONNECTOR_K = 1.0


def _fan_params(aircon: dict) -> dict | None:
    """定格 PQ が揃っていれば返す。一部だけならエラー。無ければ固定風量。"""
    present = [key for key in _FAN_KEYS if aircon.get(key) is not None]
    if not present:
        return None
    missing = [key for key in _FAN_KEYS if aircon.get(key) is None]
    if missing:
        raise ValueError(
            f"空調 {aircon.get('key', '?')}: PQ を使うときは {_FAN_KEYS} をすべて指定してください"
            f"（不足: {', '.join(missing)}）"
        )
    return {key: float(aircon[key]) for key in _FAN_KEYS}


def process_aircon(aircon: dict) -> tuple[list, list]:
    """空調を処理する"""
    nodes: list = []
    ventilation_branches: list = []

    in_node = aircon.get("in", aircon["set"])
    aircon_out_node = f"{aircon['key']}"
    out_node = aircon.get("out", aircon["set"])
    set_node = aircon["set"]
    outside_node = aircon["outside"]
    pre_temp = aircon["pre_temp"]
    pre_rh = aircon.get("pre_rh")
    model = aircon.get("model", "RAC")
    mode = aircon["mode"]
    calc_x = bool(aircon.get("calc_x", False))
    calc_c = bool(aircon.get("calc_c", False))

    fan = _fan_params(aircon)
    vol = aircon.get("vol", 1000 / 3600)

    # ノードの追加
    logger.info(
        "　エアコンを追加します: key=%s set=%s in=%s out=%s outside=%s calc_x=%s calc_c=%s airflow=%s",
        aircon_out_node,
        set_node,
        in_node,
        out_node,
        outside_node,
        calc_x,
        calc_c,
        "fan" if fan else f"vol={vol}",
    )
    logger.info(f"　エアコンノード【{aircon_out_node}】を追加します。")
    ac_node: dict = {
        "key": aircon_out_node,
        # ファンPQでは吸込ファンと吹出側圧損の交点を解くため、
        # 空調機ノードの圧力を未知数にする。固定流量では従来どおり既知境界。
        "calc_p": fan is not None,
        "calc_t": True,
        # 吸込ノード側で湿度・濃度計算を行う場合のみ、airconノードも対象にする
        "calc_x": calc_x,
        "calc_c": calc_c,
        "in_node": in_node,
        "set_node": set_node,
        "outside_node": outside_node,
        "type": "aircon",
        "pre_temp": pre_temp,
        "model": model,
        "mode": mode,
        "ac_spec": aircon.get("ac_spec", {}),
    }
    if pre_rh is not None:
        ac_node["pre_rh"] = pre_rh
    nodes.append(ac_node)

    intake_key = f"{in_node}->{aircon_out_node}"
    supply_key = f"{aircon_out_node}->{out_node}"
    if fan:
        if aircon.get("vol") is not None:
            logger.info("　空調 %s は PQ を使うため vol はファン枝に書きません。", aircon_out_node)
        logger.info(f"　換気ブランチ【{intake_key}】をファンとして追加します。")
        ventilation_branches.append(
            {
                "key": intake_key,
                "type": "fan",
                "subtype": "aircon",
                **fan,
            }
        )
        area = float(aircon.get("area") if aircon.get("area") is not None else _DEFAULT_CONNECTOR_AREA)
        k_total = float(aircon.get("k_total") if aircon.get("k_total") is not None else _DEFAULT_CONNECTOR_K)
        logger.info(f"　換気ブランチ【{supply_key}】を圧損として追加します。")
        ventilation_branches.append(
            {
                "key": supply_key,
                "type": "pressure_loss",
                "subtype": "aircon",
                "area": area,
                "k_total": k_total,
            }
        )
    else:
        for branch in (intake_key, supply_key):
            logger.info(f"　換気ブランチ【{branch}】を追加します。")
            ventilation_branches.append({"key": branch, "vol": vol, "subtype": "aircon"})

    return nodes, ventilation_branches


def process_aircons(aircons: list) -> tuple[list, list]:
    """複数の空調設定をまとめて処理する統合関数。"""
    if not aircons:
        return [], []

    nodes_all: list = []
    vents_all: list = []

    logger.info("空調の解析を開始します。")
    for ac in aircons:
        add_nodes, add_vents = process_aircon(ac)
        nodes_all.extend(add_nodes)
        vents_all.extend(add_vents)
    logger.info("空調の解析が完了しました。")

    return nodes_all, vents_all



from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)


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



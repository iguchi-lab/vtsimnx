from __future__ import annotations

from .logger import get_logger

logger = get_logger(__name__)

# [J/kg] 蒸発潜熱（簡易一定値）
_LATENT_HEAT_J_PER_KG = 2_500_000.0


def _normalize_moisture_capacity(node: dict) -> float:
    """
    入力単位を内部単位 [kg/(kg/kg)] に正規化する。

    - 既定/省略: kg/(kg/kg)
    - J/(kg/kg') 指定時: (J/(kg/kg')) / Lv[J/kg]
    """
    cap_raw = float(node["moisture_capacity"])
    unit_raw = node.get("moisture_capacity_unit", "kg/(kg/kg)")
    unit = str(unit_raw).strip().lower()

    # 既定（内部単位）
    if unit in ("kg/(kg/kg)", "kg_per_kgkg", "kgkg"):
        return cap_raw

    # エネルギー基準（湿度比1あたり）
    if unit in ("j/(kg/kg')", "j/(kg/kg)", "j_per_kgkg", "jkgkg"):
        cap_conv = cap_raw / _LATENT_HEAT_J_PER_KG
        logger.info(
            "　湿気容量を変換します: key=%s %.6g [J/(kg/kg')] -> %.6g [kg/(kg/kg)]",
            node.get("key", "?"),
            cap_raw,
            cap_conv,
        )
        return cap_conv

    raise ValueError(
        f"Unsupported moisture_capacity_unit={unit_raw!r} for node={node.get('key', '?')}. "
        "Use 'kg/(kg/kg)' or 'J/(kg/kg\\')'."
    )


def add_moisture_capacity(node: dict, time_step: float) -> tuple[list, list]:
    """
    材料側の湿気容量ノードと湿気伝達ブランチを追加する。
    - ノード: <key>_mx（calc_x=true）
    - ブランチ: <key>_mx-><key>（moisture_conductance = moisture_capacity / dt）
    """
    nodes: list = []
    thermal_branches: list = []

    key = str(node["key"])
    cap = _normalize_moisture_capacity(node)
    init_x = node.get("x", 0.0)

    logger.info("　湿気容量ノード【%s_mx】を追加します。", key)
    nodes.append(
        {
            "key": f"{key}_mx",
            "calc_x": True,
            "calc_t": False,
            "type": "capacity",
            "subtype": "moisture",
            "ref_node": key,
            "x": init_x,
            "moisture_capacity": cap,
        }
    )

    logger.info("　湿気伝達ブランチ【%s_mx->%s】を追加します。", key, key)
    thermal_branches.append(
        {
            "key": f"{key}_mx->{key}",
            "source": f"{key}_mx",
            "target": key,
            "type": "conductance",
            "subtype": "moisture_capacity",
            "conductance": 0.0,
            "moisture_conductance": cap / time_step,
        }
    )
    return nodes, thermal_branches


def process_moisture_capacities(node_config: list, time_step: float) -> tuple[list, list]:
    """
    ノード配列を走査し、moisture_capacity を持つノードに
    湿気容量ノード/湿気ブランチを付与する。
    """
    add_nodes_all: list = []
    add_tb_all: list = []
    logger.info("湿気容量を追加します")
    for node in node_config:
        if "moisture_capacity" in node:
            node["calc_x"] = True
            add_nodes, add_tb = add_moisture_capacity(node, time_step)
            add_nodes_all.extend(add_nodes)
            add_tb_all.extend(add_tb)
            node.pop("moisture_capacity", None)
            node.pop("moisture_capacity_unit", None)
    logger.info("湿気容量の追加が完了しました。")
    return add_nodes_all, add_tb_all


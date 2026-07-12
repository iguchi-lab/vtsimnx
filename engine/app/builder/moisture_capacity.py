from __future__ import annotations

import math

from .logger import get_logger

logger = get_logger(__name__)

# [J/kg] 蒸発潜熱（簡易一定値）
_LATENT_HEAT_J_PER_KG = 2_500_000.0


def _require_positive_finite(value: float, *, name: str, context: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context}: {name} must be finite and > 0, got {value!r}")
    return value


def _normalize_moisture_capacity(node: dict) -> float:
    """
    入力単位を内部単位 [kg/(kg/kg)] に正規化する。

    - 既定/省略: J/(kg/kg')
    - kg/(kg/kg) 指定時: そのまま
    - J/(kg/kg') 指定時: (J/(kg/kg')) / Lv[J/kg]
    """
    key = node.get("key", "?")
    try:
        cap_raw = float(node["moisture_capacity"])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"node {key}: moisture_capacity must be a finite number > 0, got {node.get('moisture_capacity')!r}"
        ) from e
    _require_positive_finite(cap_raw, name="moisture_capacity", context=f"node {key}")

    unit_raw = node.get("moisture_capacity_unit", "J/(kg/kg')")
    unit = str(unit_raw).strip().lower()

    # 既定（内部単位）
    if unit in ("kg/(kg/kg)", "kg_per_kgkg", "kgkg"):
        return cap_raw

    # エネルギー基準（湿度比1あたり）
    if unit in ("j/(kg/kg')", "j/(kg/kg)", "j_per_kgkg", "jkgkg"):
        cap_conv = cap_raw / _LATENT_HEAT_J_PER_KG
        logger.info(
            "　湿気容量を変換します: key=%s %.6g [J/(kg/kg')] -> %.6g [kg/(kg/kg)]",
            key,
            cap_raw,
            cap_conv,
        )
        return _require_positive_finite(
            cap_conv, name="moisture_capacity (converted)", context=f"node {key}"
        )

    raise ValueError(
        f"Unsupported moisture_capacity_unit={unit_raw!r} for node={key}. "
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
    try:
        dt = float(time_step)
    except (TypeError, ValueError) as e:
        raise ValueError(f"timestep must be finite and > 0, got {time_step!r}") from e
    _require_positive_finite(dt, name="timestep", context="moisture_capacity expansion")

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
            "moisture_conductance": cap / dt,
        }
    )
    return nodes, thermal_branches


def derive_calc_x_from_moisture_capacity(node_config: list) -> None:
    """
    湿気容量展開前に、moisture_capacity を持つノードへ calc_x=True を立てる。

    add_moisture_capacity=True のときだけ呼ぶこと（空調ノードへの伝播より先）。
    """
    for node in node_config:
        if isinstance(node, dict) and "moisture_capacity" in node:
            node["calc_x"] = True


def strip_moisture_capacity_fields(node_config: list) -> None:
    """
    add_moisture_capacity=False のとき、solver へ直接渡さないよう moisture_capacity を除去する。

    False は「材料側ノードへ展開しない別モデル」ではなく「湿気容量を無効」と解釈する。
    moisture_capacity / moisture_capacity_unit のどちらか一方でもあれば両方除去する。
    """
    for node in node_config:
        if not isinstance(node, dict):
            continue
        if "moisture_capacity" not in node and "moisture_capacity_unit" not in node:
            continue
        logger.info(
            "　湿気容量を無効化します（add_moisture_capacity=False）: key=%s",
            node.get("key", "?"),
        )
        node.pop("moisture_capacity", None)
        node.pop("moisture_capacity_unit", None)


def process_moisture_capacities(node_config: list, time_step: float) -> tuple[list, list]:
    """
    ノード配列を走査し、moisture_capacity を持つノードに
    湿気容量ノード/湿気ブランチを付与する。
    """
    add_nodes_all: list = []
    add_tb_all: list = []
    logger.info("湿気容量を追加します")
    for node in node_config:
        if not isinstance(node, dict):
            continue
        has_cap = "moisture_capacity" in node
        has_unit = "moisture_capacity_unit" in node
        if has_unit and not has_cap:
            raise ValueError(
                f"node {node.get('key', '?')}: moisture_capacity_unit without moisture_capacity"
            )
        if has_cap:
            node["calc_x"] = True
            add_nodes, add_tb = add_moisture_capacity(node, time_step)
            add_nodes_all.extend(add_nodes)
            add_tb_all.extend(add_tb)
            node.pop("moisture_capacity", None)
            node.pop("moisture_capacity_unit", None)
    logger.info("湿気容量の追加が完了しました。")
    return add_nodes_all, add_tb_all

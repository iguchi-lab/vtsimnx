from __future__ import annotations

from typing import Any

from .logger import get_logger
from .utils import _normalize_timeseries_mapping

logger = get_logger(__name__)

_FAN_KEYS = ("p_max", "p1", "q1", "q_max")
_DEFAULT_CONNECTOR_AREA = 0.05
_DEFAULT_CONNECTOR_K = 1.0
_DEFAULT_VOL = 150.0 / 3600.0


def parse_heat_recovery_vents(raw: dict[str, Any]) -> list[dict[str, Any]]:
    items = raw.get("heat_recovery_vent")
    if items is None:
        logger.info("熱交換換気の設定が見つかりませんでした。")
        return []
    if not isinstance(items, list):
        raise TypeError("heat_recovery_vent セクションは list である必要があります。")

    logger.info("熱交換換気の解析を開始します")
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"不正な熱交換換気定義: {item!r}")
        out.append(_normalize_timeseries_mapping(item))
    return out


def _fan_params(block: dict[str, Any] | None, *, label: str) -> dict[str, float] | None:
    if not isinstance(block, dict):
        return None
    present = [key for key in _FAN_KEYS if block.get(key) is not None]
    if not present:
        return None
    missing = [key for key in _FAN_KEYS if block.get(key) is None]
    if missing:
        raise ValueError(
            f"熱交換換気 {label}: PQ を使うときは {_FAN_KEYS} をすべて指定してください"
            f"（不足: {', '.join(missing)}）"
        )
    return {key: float(block[key]) for key in _FAN_KEYS}


def _recovery_mode(item: dict[str, Any]) -> str:
    recovery = str(item.get("recovery", item.get("model", "sensible"))).strip().lower()
    if recovery in ("sensible", "顕熱", "hrv"):
        return "sensible"
    if recovery in ("total", "enthalpy", "全熱", "erv"):
        return "total"
    raise ValueError(
        f"熱交換換気 {item.get('key', '?')}: recovery は sensible/total のいずれかです（got {recovery!r}）"
    )


def _eta(item: dict[str, Any], key: str, default: float) -> float:
    if item.get(key) is None:
        return default
    try:
        v = float(item[key])
    except Exception as e:
        raise ValueError(f"熱交換換気 {item.get('key', '?')}: {key} は数値である必要があります") from e
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"熱交換換気 {item.get('key', '?')}: {key} は 0..1 です（got {v}）")
    return v


def _first_str(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if item.get(key) is not None and str(item[key]).strip() != "":
            return str(item[key])
    return None


def _resolve_ports(item: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    OA/SA/RA/EA ポートを解決する。

    短縮形:
      outdoor + room → oa=ea=outdoor, sa=ra=room
    個別指定:
      oa / sa / ra / ea（別名あり）を優先
      ※ supply/exhaust はファン PQ 用のためポート名には使わない
    """
    key = str(item.get("key", "?"))
    outdoor = _first_str(item, "oa", "outdoor")
    room = _first_str(item, "room", "set")

    oa = _first_str(item, "oa", "outdoor")
    sa = _first_str(item, "sa", "out")
    ra = _first_str(item, "ra", "in", "return", "room", "set")
    ea = _first_str(item, "ea", "exhaust_out")

    if oa is None and outdoor is not None:
        oa = outdoor
    if ea is None and outdoor is not None:
        ea = outdoor
    if sa is None and room is not None:
        sa = room
    if ra is None and room is not None:
        ra = room

    missing = [name for name, val in (("oa", oa), ("sa", sa), ("ra", ra), ("ea", ea)) if val is None]
    if missing:
        raise ValueError(
            f"熱交換換気 {key}: ポート {', '.join(missing)} を指定してください"
            f"（例: oa/sa/ra/ea、または outdoor+room）"
        )
    assert oa is not None and sa is not None and ra is not None and ea is not None
    return oa, sa, ra, ea


def process_heat_recovery_vent(item: dict[str, Any]) -> tuple[list, list]:
    """熱交換換気 1 台を nodes / ventilation_branches に展開する。"""
    key = str(item["key"])
    oa, sa, ra, ea = _resolve_ports(item)

    recovery = _recovery_mode(item)
    eta_t = _eta(item, "eta_t", 0.7)
    eta_x = _eta(item, "eta_x", 0.0 if recovery == "sensible" else 0.6)
    if recovery == "sensible":
        eta_x = 0.0

    supply_fan = _fan_params(item.get("supply"), label=f"{key}.supply")
    exhaust_fan = _fan_params(item.get("exhaust"), label=f"{key}.exhaust")
    if (supply_fan is None) != (exhaust_fan is None):
        raise ValueError(
            f"熱交換換気 {key}: supply/exhaust の PQ は両方指定するか、両方省略（vol）してください"
        )
    use_fan = supply_fan is not None
    vol = float(item["vol"]) if item.get("vol") is not None else _DEFAULT_VOL

    # 機器内ノード（給気側境界 / 排気側ジャンクション）
    hrv_sa_node = key
    hrv_ea_node = f"{key}_exhaust"
    nodes: list = []
    vents: list = []

    logger.info(
        "　熱交換換気を追加します: key=%s oa=%s sa=%s ra=%s ea=%s recovery=%s eta_t=%s eta_x=%s airflow=%s",
        key,
        oa,
        sa,
        ra,
        ea,
        recovery,
        eta_t,
        eta_x,
        "fan" if use_fan else f"vol={vol}",
    )

    nodes.append(
        {
            "key": hrv_sa_node,
            "type": "hrv",
            "calc_p": use_fan,
            "calc_t": False,
            "calc_x": False,
            "calc_c": False,
            # 熱交換の参照: OA / RA（SA 境界は本ノード自身）
            "outside_node": oa,
            "in_node": ra,
            "set_node": sa,
            "model": recovery,
            "ac_spec": {
                "eta_t": eta_t,
                "eta_x": eta_x,
                "recovery": recovery,
                "oa_node": oa,
                "sa_node": sa,
                "ra_node": ra,
                "ea_node": ea,
                "exhaust_node": hrv_ea_node,
            },
            "t": 20.0,
            "x": 0.0,
        }
    )
    nodes.append(
        {
            "key": hrv_ea_node,
            "type": "normal",
            "subtype": "hrv_exhaust",
            "calc_p": use_fan,
            "calc_t": False,
            "calc_x": False,
            "calc_c": False,
            "t": 20.0,
            "x": 0.0,
        }
    )

    # OA → HRV給気 → SA
    # RA → HRV排気 → EA
    oa_to_hrv = f"{oa}->{hrv_sa_node}"
    hrv_to_sa = f"{hrv_sa_node}->{sa}"
    ra_to_hrv = f"{ra}->{hrv_ea_node}"
    hrv_to_ea = f"{hrv_ea_node}->{ea}"

    if use_fan:
        if item.get("vol") is not None:
            logger.info("　熱交換換気 %s は PQ を使うため vol はファン枝に書きません。", key)
        area = float(item["area"]) if item.get("area") is not None else _DEFAULT_CONNECTOR_AREA
        k_total = float(item["k_total"]) if item.get("k_total") is not None else _DEFAULT_CONNECTOR_K
        vents.append(
            {
                "key": oa_to_hrv,
                "type": "fan",
                "subtype": "hrv",
                **supply_fan,
            }
        )
        vents.append(
            {
                "key": hrv_to_sa,
                "type": "pressure_loss",
                "subtype": "hrv",
                "area": area,
                "k_total": k_total,
            }
        )
        vents.append(
            {
                "key": ra_to_hrv,
                "type": "fan",
                "subtype": "hrv",
                **exhaust_fan,
            }
        )
        vents.append(
            {
                "key": hrv_to_ea,
                "type": "pressure_loss",
                "subtype": "hrv",
                "area": area,
                "k_total": k_total,
            }
        )
    else:
        for branch in (oa_to_hrv, hrv_to_sa, ra_to_hrv, hrv_to_ea):
            vents.append({"key": branch, "vol": vol, "subtype": "hrv"})

    return nodes, vents


def process_heat_recovery_vents(items: list) -> tuple[list, list]:
    if not items:
        return [], []
    nodes: list = []
    vents: list = []
    for item in items:
        add_nodes, add_vents = process_heat_recovery_vent(item)
        nodes.extend(add_nodes)
        vents.extend(add_vents)
    return nodes, vents

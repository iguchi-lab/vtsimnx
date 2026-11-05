"""計算は通るが、モデル意図として疑わしい入力の警告。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_FLOW_IMBALANCE_M3S = 1e-9
_SKIP_NODE_KEYS = frozenset({"void"})


def _issue(
    *,
    code: str,
    message: str,
    path: List[Any],
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "code": code,
        "message": message,
        "path": path,
        "severity": "warning",
    }
    if hint:
        out["hint"] = hint
    return out


def _split_arrow(key: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(key, str) or "->" not in key:
        return None, None
    left, right = key.split("->", 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None, None
    return left, right


def _endpoints(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    src = item.get("source")
    dst = item.get("target")
    if isinstance(src, str) and src and isinstance(dst, str) and dst:
        return src, dst
    key = item.get("key")
    if isinstance(key, str):
        return _split_arrow(key)
    return None, None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        try:
            return max(abs(float(x)) for x in value if isinstance(x, (int, float)))
        except (TypeError, ValueError):
            return None
    return None


def _mentioned_nodes(raw: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for section in ("ventilation_branches", "thermal_branches"):
        for item in raw.get(section) or []:
            if not isinstance(item, dict):
                continue
            src, dst = _endpoints(item)
            if src:
                keys.add(src)
            if dst:
                keys.add(dst)
    for surface in raw.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        src, dst = _split_arrow(str(surface.get("key") or ""))
        if src:
            keys.add(src)
        if dst:
            keys.add(dst)
    for ac in raw.get("aircon") or []:
        if not isinstance(ac, dict):
            continue
        for field in ("set", "in", "out", "outside"):
            val = ac.get(field)
            if isinstance(val, str) and val:
                keys.add(val)
    for hrv in raw.get("heat_recovery_vent") or []:
        if not isinstance(hrv, dict):
            continue
        for field in ("outdoor", "room", "set", "oa", "sa", "ra", "ea", "in", "out"):
            val = hrv.get(field)
            if isinstance(val, str) and val:
                keys.add(val)
    for src in raw.get("heat_source") or []:
        if isinstance(src, dict):
            room = src.get("room")
            if isinstance(room, str) and room:
                keys.add(room)
    for src in raw.get("humidity_source") or []:
        if isinstance(src, dict):
            room = src.get("room") or src.get("target_node") or src.get("set")
            if isinstance(room, str) and room:
                keys.add(room)
    return keys


def lint_raw_config(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """raw_config（prepare 後）を見て、意図のずれを警告する。エラーにはしない。"""
    issues: List[Dict[str, Any]] = []
    nodes = raw.get("nodes") or []
    node_index = {
        str(n.get("key")): i
        for i, n in enumerate(nodes)
        if isinstance(n, dict) and n.get("key") is not None
    }
    node_by_key = {
        str(n.get("key")): n
        for n in nodes
        if isinstance(n, dict) and n.get("key") is not None
    }

    net_flow: Dict[str, float] = {}
    for i, branch in enumerate(raw.get("ventilation_branches") or []):
        if not isinstance(branch, dict):
            continue
        btype = str(branch.get("type") or "fixed_flow")
        if btype != "fixed_flow":
            continue
        vol = _as_float(branch.get("vol"))
        if vol is None or abs(vol) <= _FLOW_IMBALANCE_M3S:
            continue
        src, dst = _endpoints(branch)
        if not src or not dst or src == "void" or dst == "void":
            continue
        signed = float(branch.get("vol") if isinstance(branch.get("vol"), (int, float)) else vol)
        net_flow[src] = net_flow.get(src, 0.0) - signed
        net_flow[dst] = net_flow.get(dst, 0.0) + signed

    for key, net in sorted(net_flow.items()):
        if key in _SKIP_NODE_KEYS or abs(net) <= _FLOW_IMBALANCE_M3S:
            continue
        idx = node_index.get(key)
        path: List[Any] = ["nodes", idx] if idx is not None else ["ventilation_branches"]
        issues.append(
            _issue(
                code="fixed_flow_imbalance",
                message=(
                    f"ノード '{key}' の固定流量の出入りが釣り合っていません"
                    f"（正味 {net:.6g} m³/s）。給気だけ、または排気だけの設定になっていないか確認してください。"
                ),
                path=path,
                hint="還気・排気の fixed_flow を追加するか、開口/圧損枝でバランスを取ってください。",
            )
        )

    mentioned = _mentioned_nodes(raw)
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        key = str(node.get("key") or "")
        if not key or key in _SKIP_NODE_KEYS:
            continue
        calc_t = bool(node.get("calc_t"))
        volume = node.get("v")
        has_volume = isinstance(volume, (int, float)) and not isinstance(volume, bool) and float(volume) > 0.0
        has_mass = isinstance(node.get("thermal_mass"), (int, float)) and float(node.get("thermal_mass") or 0) > 0.0
        if calc_t and not has_volume and not has_mass:
            issues.append(
                _issue(
                    code="missing_room_capacity",
                    message=f"ノード '{key}' は calc_t=true ですが気積 v も thermal_mass もありません。",
                    path=["nodes", i, "v"],
                    hint="室の空気熱容量のため v [m³] を指定するか、家具等の thermal_mass [J/K] を指定してください。",
                )
            )
        if key not in mentioned:
            issues.append(
                _issue(
                    code="isolated_node",
                    message=f"ノード '{key}' は換気枝・熱枝・壁・空調のいずれにも接続されていません。",
                    path=["nodes", i, "key"],
                    hint="ventilation_branches / thermal_branches / surfaces / aircon のいずれかから参照してください。",
                )
            )

    for i, src in enumerate(raw.get("humidity_source") or []):
        if not isinstance(src, dict):
            continue
        room = src.get("room") or src.get("target_node") or src.get("set")
        if not isinstance(room, str) or not room:
            continue
        node = node_by_key.get(room)
        if node is not None and not bool(node.get("calc_x")):
            idx = node_index.get(room)
            issues.append(
                _issue(
                    code="humidity_source_without_calc_x",
                    message=f"発湿先 '{room}' に calc_x が立っていません。湿度計算に反映されないことがあります。",
                    path=["humidity_source", i, "room"],
                    hint="対象室ノードに calc_x=true を指定してください（builder が後で立てる場合もあります）。",
                )
            )

    for i, ac in enumerate(raw.get("aircon") or []):
        if not isinstance(ac, dict):
            continue
        for field, hint in (
            ("set", "温度制御対象の室ノード"),
            ("outside", "外気ノード"),
            ("pre_temp", "設定温度"),
            ("mode", "運転モード（HEATING/COOLING/OFF 等）"),
        ):
            if ac.get(field) is None:
                issues.append(
                    _issue(
                        code="aircon_missing_field",
                        message=f"aircon[{i}] に '{field}' がありません。",
                        path=["aircon", i, field],
                        hint=hint,
                    )
                )

    return issues


def lint_built_config(built: Dict[str, Any]) -> List[Dict[str, Any]]:
    """展開後 JSON の補助警告。"""
    issues: List[Dict[str, Any]] = []
    flags = (built.get("simulation") or {}).get("calc_flag") or {}
    if not any(bool(flags.get(k)) for k in ("p", "t", "x", "c")):
        issues.append(
            _issue(
                code="no_calc_flag",
                message="展開後の calc_flag がすべて false です。未知数のない計算になっています。",
                path=["simulation", "calc_flag"],
                hint="室ノードに calc_t / calc_p / calc_x / calc_c のいずれかを指定してください。",
            )
        )
    return issues

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Iterable
import re
import numpy as np
import pandas as pd

# ------------------------------
# 定数（区切り文字）
# ------------------------------
CHAIN_DELIMITER = "->"      # ノード連鎖の区切り
COMMENT_DELIMITER = "||"    # インラインコメントの区切り
COMPOUND_DELIMITER = "&&"   # 複合キー（AND条件）の区切り

_INT_RE   = re.compile(r'^[+-]?\d+\Z')
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')


# ------------------------------
# ユーティリティ
# ------------------------------
def _split_key_and_comment(key: str) -> Tuple[str, str]:
    k = key.strip()
    if COMMENT_DELIMITER in k:
        head, tail = k.split(COMMENT_DELIMITER, 1)
        return head.strip(), tail.strip()
    return k, ""


def _split_compound_key(key: str, delimiter: str = COMPOUND_DELIMITER) -> List[str]:
    k = key.strip()
    if delimiter in k:
        return [part.strip() for part in k.split(delimiter)]
    return [k]


def _expand_chain(key: str) -> List[str]:
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
    merged = {**base, **overrides}
    return _normalize_timeseries_mapping(merged)


def convert_numeric_values(
    obj: Any,
    *,
    bool_keys: Optional[Iterable[str]] = None,
    _parent_key: Optional[str] = None,
) -> Any:
    bool_keys_set = set(bool_keys or ())

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            converted = convert_numeric_values(v, bool_keys=bool_keys_set, _parent_key=str(k))
            if (
                str(k) in bool_keys_set
                and isinstance(converted, (int, np.integer))
                and converted in (0, 1)
            ):
                out[k] = bool(converted)
            else:
                out[k] = converted
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return obj
        converted = [convert_numeric_values(x, bool_keys=bool_keys_set, _parent_key=_parent_key) for x in obj]
        if all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
               for x in converted):
            return np.array(converted)
        return converted if isinstance(obj, list) else tuple(converted)

    if isinstance(obj, str):
        s = obj.strip()
        if _INT_RE.match(s):
            return int(s)
        if _FLOAT_RE.match(s):
            return float(s)
        return obj

    if isinstance(obj, (bool, np.bool_)):
        return obj

    if isinstance(obj, (int, float, np.integer, np.floating)):
        return obj

    return obj


def convert_to_json_compatible(obj: Any) -> Any:
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
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    return [value] * length



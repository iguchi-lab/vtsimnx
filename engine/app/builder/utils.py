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


def series_summary_for_log(value: Any) -> str:
    """
    ログ用の時系列サマリ。配列は len / 先頭・末尾のみ表示する。
    heat_sources / moisture などで共通利用。
    """
    if isinstance(value, list):
        if not value:
            return "series[len=0]"
        return f"series[len={len(value)} first={value[0]} last={value[-1]}]"
    return f"scalar[{value}]"


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
    """
    時系列を sim length に正規化する。

    - スカラー / 0次元 ndarray: length まで展開
    - 長さ 1 の配列: length まで展開
    - 長さ length: そのまま
    - それ以外（空・不一致長）: ValueError
    """
    if length <= 0:
        raise ValueError(f"ensure_timeseries: length must be positive, got {length}")

    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [value.item()] * length

    if isinstance(value, (list, tuple, np.ndarray)):
        seq = list(value)
        n = len(seq)
        if n == length:
            return seq
        if n == 1:
            return [seq[0]] * length
        raise ValueError(
            f"timeseries length mismatch: got {n}, expected {length} (or 1 to broadcast)"
        )

    return [value] * length


def _is_sequence_like(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def normalize_optional_series(
    obj: dict,
    field: str,
    *,
    length: int,
    default=None,
    fill_if_missing: bool = False,
    expand_scalars: bool = True,
) -> None:
    """
    obj[field] を時系列長 length に正規化する（破壊的）。

    - 欠落時: fill_if_missing なら default を length 展開
    - 存在時: ensure_timeseries で正規化
    - expand_scalars=False: スカラー（非シーケンス）はそのまま残す（巨大化回避用）
    """
    if field not in obj or obj[field] is None:
        if fill_if_missing:
            obj[field] = ensure_timeseries(default, length)
        return

    value = obj[field]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        if expand_scalars:
            obj[field] = ensure_timeseries(value, length)
        else:
            obj[field] = value.item()
        return

    if not expand_scalars and not _is_sequence_like(value):
        return

    obj[field] = ensure_timeseries(value, length)




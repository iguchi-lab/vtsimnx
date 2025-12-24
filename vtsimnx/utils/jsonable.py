from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _is_nan_or_inf(x: float) -> bool:
    # JSONとしてInfinity/NaNを送ると相手側で失敗することがあるため None に落とす
    return math.isnan(x) or math.isinf(x)


def to_jsonable(obj: Any) -> Any:
    """
    JSON化できない型（pandas/numpy 等）を、JSON互換の型へ再帰変換する。

    - pd.Series -> list
    - pd.DataFrame -> dict[str, list]
    - numpy scalar/ndarray -> python scalar / list
    - Timestamp/datetime/date -> ISO文字列
    - NaN/Inf/NaT -> None
    """
    # None / bool / int / str はそのまま
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # float: NaN/Inf を None へ
    if isinstance(obj, float):
        return None if _is_nan_or_inf(obj) else obj

    # datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # pandas
    if isinstance(obj, pd.Timestamp):
        # NaT もここに来る可能性があるため isna を見る
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()

    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, pd.Index):
        return [to_jsonable(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        # orient="list" で列->配列の形にしてから再帰変換
        as_dict = obj.to_dict(orient="list")
        return {str(k): [to_jsonable(v) for v in vals] for k, vals in as_dict.items()}

    # numpy（依存は環境によっては optional の可能性があるので遅延import）
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return to_jsonable(obj.item())

        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    except Exception:
        pass

    # dict / list / tuple
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # JSONのキーは文字列のみ
            out[str(k)] = to_jsonable(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # pandas の NaT / numpy.nan などスカラ判定の最後の砦
    try:
        if pd.isna(obj):  # type: ignore[arg-type]
            return None
    except Exception:
        pass

    raise TypeError(f"to_jsonable: JSONに変換できない型です: {type(obj).__name__}")


__all__ = ["to_jsonable"]



"""schema / f32.bin からの DataFrame 復元。"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from vtsimnx.units import unit_for_series

from .errors import ArtifactDecodeError
from ._schema import series_columns


def load_json_bytes(raw: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ArtifactDecodeError(f"invalid json bytes: {e}") from e
    if not isinstance(obj, dict):
        raise ArtifactDecodeError(f"expected JSON object, got {type(obj).__name__}")
    return obj


def validate_f32_schema(schema: Dict[str, Any]) -> int:
    """dtype/layout/length を検証し length(T) を返す。"""
    dtype = schema.get("dtype")
    layout = schema.get("layout")
    if dtype != "f32le":
        raise ArtifactDecodeError(f"schema.json dtype が想定外です: {dtype!r} (想定: 'f32le')")
    if layout != "timestep-major":
        raise ArtifactDecodeError(f"schema.json layout が想定外です: {layout!r} (想定: 'timestep-major')")
    T = schema.get("length")
    if not isinstance(T, int) or T < 0:
        raise ArtifactDecodeError(f"schema.json length が不正です: {T!r}")
    return T


def apply_time_index(
    df: pd.DataFrame,
    index_spec: Dict[str, Any],
    *,
    expected_length: int,
) -> None:
    """index_spec（start/timestep/length）に基づいて df.index を time 軸にする。"""
    start = index_spec.get("start")
    timestep = index_spec.get("timestep")
    length = index_spec.get("length")
    if not (isinstance(start, str) and isinstance(timestep, int) and isinstance(length, int)):
        return
    if length != expected_length:
        return

    start_ts = pd.to_datetime(start)
    if timestep == 0:
        df.index = pd.DatetimeIndex([start_ts] * expected_length)
    else:
        df.index = pd.date_range(
            start=start_ts, periods=expected_length, freq=pd.to_timedelta(timestep, unit="s")
        )
    df.index.name = "time"


def attach_series_units(df: pd.DataFrame, series_name: str) -> None:
    """vtsimnx.units.SERIES_UNITS があれば DataFrame.attrs に付与する。"""
    unit = unit_for_series(series_name)
    if unit is None:
        return
    df.attrs["unit"] = unit
    df.attrs["series"] = series_name


def decode_f32_series(
    raw: bytes,
    schema: Dict[str, Any],
    series_name: str,
    *,
    index_spec: Optional[Dict[str, Any]] = None,
    attach_units: bool = True,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    f32le timestep-major バイナリを DataFrame に復元する（HTTP 非依存）。

    tools / run_calc / get_artifact_file の共通実装。
    """
    T = validate_f32_schema(schema)
    cols = series_columns(schema, series_name)
    N = len(cols)

    label = source_name or series_name
    try:
        arr = np.frombuffer(raw, dtype=np.dtype("<f4"))
    except ValueError as e:
        raise ArtifactDecodeError(f"{label}: f32 バイナリとして読めません: {e}") from e
    expected = T * N
    if arr.size != expected:
        raise ArtifactDecodeError(
            f"{label}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})"
        )
    arr = arr.reshape((T, N))
    df = pd.DataFrame(arr, columns=cols)

    if isinstance(index_spec, dict):
        try:
            apply_time_index(df, index_spec, expected_length=T)
        except (TypeError, ValueError):
            # index 付与失敗でも DataFrame 自体は返す
            pass

    if attach_units:
        attach_series_units(df, series_name)
    return df


__all__ = [
    "load_json_bytes",
    "validate_f32_schema",
    "apply_time_index",
    "attach_series_units",
    "decode_f32_series",
]

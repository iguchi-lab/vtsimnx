from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from vtsimnx.artifacts._decode import build_time_index

from ._response import _output_block


def _index_spec_from_config(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict):
        return None
    sim = config.get("simulation")
    if not isinstance(sim, dict):
        return None
    spec = sim.get("index")
    return spec if isinstance(spec, dict) else None


def _index_spec_from_output(resp_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    output = _output_block(resp_json)
    spec = output.get("index")
    return spec if isinstance(spec, dict) else None


def _time_index_from_spec(spec: Dict[str, Any], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """index spec dict から DatetimeIndex を復元（正本は artifacts.build_time_index）。"""
    return build_time_index(spec, expected_length=expected_length)


def _time_index_from_config(config: Optional[Dict[str, Any]], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """
    config["simulation"]["index"] が dict（start/end/timestep/length）なら DatetimeIndex を復元する。
    expected_length と length が一致しない場合は None。
    """
    spec = _index_spec_from_config(config)
    if spec is None:
        return None
    return build_time_index(spec, expected_length=expected_length)


def _time_index_from_output(resp_json: Dict[str, Any], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """
    APIレスポンス（/runのJSON）に含まれる output.index から DatetimeIndex を復元する。
    expected_length と length が一致しない場合は None。
    """
    spec = _index_spec_from_output(resp_json)
    if spec is None:
        return None
    return build_time_index(spec, expected_length=expected_length)


def _pick_index_spec(
    resp_json: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    *,
    expected_length: int,
) -> Optional[Dict[str, Any]]:
    """output.index を優先し、長さ不一致なら config の index へフォールバック。"""
    for spec in (_index_spec_from_output(resp_json), _index_spec_from_config(config)):
        if spec is None:
            continue
        if build_time_index(spec, expected_length=expected_length) is not None:
            return spec
    return None


def _warmup_eval_length(sim: Dict[str, Any], warmup: Dict[str, Any]) -> Optional[int]:
    if isinstance(warmup.get("eval_length"), int):
        return int(warmup["eval_length"])
    idx = sim.get("index")
    if isinstance(idx, dict) and isinstance(idx.get("length"), int):
        return int(idx["length"])
    if idx is None or isinstance(idx, (str, bytes, dict)):
        return None
    n = getattr(idx, "__len__", None)
    if callable(n):
        return int(n())
    return None


def _warmup_steps_from_start(sim: Dict[str, Any], start_spec: str) -> Optional[int]:
    """典型年の MM-DD 以降が先頭に付くステップ数。index にその日が無いときは None。"""
    spec = sim.get("index")
    if not isinstance(spec, dict):
        return None
    start = spec.get("start")
    timestep = spec.get("timestep")
    length = spec.get("length")
    if not isinstance(start, str) or not start:
        return None
    if not isinstance(timestep, int) or timestep <= 0:
        return None
    if not isinstance(length, int) or length < 2:
        return None
    parts = str(start_spec).replace("/", "-").strip().split("-")
    if len(parts) == 3:
        month, day = int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        month, day = int(parts[0]), int(parts[1])
    else:
        return None
    idx = pd.date_range(
        start=pd.to_datetime(start),
        periods=length,
        freq=pd.to_timedelta(int(timestep), unit="s"),
    )
    first = idx[0]
    if (int(first.month), int(first.day)) >= (month, day):
        return None
    mask = (idx.month > month) | ((idx.month == month) & (idx.day >= day))
    steps = int(mask.sum())
    return steps if steps > 0 else None


def _warmup_skip_rows(config: Optional[Dict[str, Any]], schema_length: int) -> int:
    """助走つき結果なら、評価期間より長い先頭を落とす行数。"""
    if not isinstance(config, dict):
        return 0
    sim = config.get("simulation")
    if not isinstance(sim, dict):
        return 0
    warmup = sim.get("warmup")
    if isinstance(warmup, str):
        warmup = {"start": warmup}
    if not isinstance(warmup, dict):
        return 0
    if not warmup.get("start") and not warmup.get("steps") and not warmup.get("eval_length"):
        return 0

    eval_len = _warmup_eval_length(sim, warmup)
    steps = warmup.get("steps")
    if isinstance(steps, int) and steps > 0:
        if eval_len is not None and schema_length == eval_len + steps:
            return steps
        if eval_len is None and steps < schema_length:
            return steps
    if eval_len is not None and schema_length > eval_len:
        return schema_length - eval_len

    start_spec = warmup.get("start")
    if isinstance(start_spec, str) and start_spec:
        computed = _warmup_steps_from_start(sim, start_spec)
        if computed is not None and schema_length == (eval_len or 0) + computed:
            return computed
        if computed is not None and eval_len is None and schema_length > computed:
            return computed
    return 0


def _normalize_simulation_index_inplace(cfg: Dict[str, Any]) -> None:
    """
    cfg["simulation"]["index"] が DatetimeIndex（または datetime 配列）なら
    API互換の dict 形式へ正規化する:
        {"start": "...", "end": "...", "timestep": 3600, "length": 8760}
    """
    sim = cfg.get("simulation")
    if not isinstance(sim, dict):
        return

    idx = sim.get("index")
    # すでに dict なら何もしない
    if isinstance(idx, dict):
        return

    # pandas DatetimeIndex / Index / list などを DatetimeIndex へ寄せる
    try:
        if isinstance(idx, pd.Index):
            dt_index = pd.DatetimeIndex(idx)
        elif isinstance(idx, (list, tuple)):
            dt_index = pd.to_datetime(list(idx))
        else:
            return
    except Exception:
        return

    if len(dt_index) == 0:
        return

    # timestep 推定（一定間隔前提）。1点しかないなら 0 とする
    if len(dt_index) >= 2:
        deltas = np.diff(dt_index.asi8)  # ns
        step_ns = int(deltas[0])
        if not np.all(deltas == step_ns):
            raise ValueError("simulation.index の間隔が一定ではありません（timestep を推定できません）。")
        timestep = int(round(step_ns / 1_000_000_000))
    else:
        timestep = 0

    def _fmt(ts: pd.Timestamp) -> str:
        # API側の既存例に合わせて "YYYY-MM-DD HH:MM:SS" 形式にする
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.strftime("%Y-%m-%d %H:%M:%S")

    sim["index"] = {
        "start": _fmt(dt_index[0]),
        "end": _fmt(dt_index[-1]),
        "timestep": timestep,
        "length": int(len(dt_index)),
    }


__all__ = [
    "_normalize_simulation_index_inplace",
    "_pick_index_spec",
    "_index_spec_from_config",
    "_index_spec_from_output",
    "_time_index_from_config",
    "_time_index_from_output",
    "_warmup_skip_rows",
]

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ._response import _output_block


def _time_index_from_spec(spec: Dict[str, Any], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """
    index spec dict（start/end/timestep/length）から DatetimeIndex を復元する。
    expected_length と length が一致しない場合は None。
    """
    start = spec.get("start")
    timestep = spec.get("timestep")
    length = spec.get("length")
    if not isinstance(start, str) or not start:
        return None
    if not isinstance(timestep, int) or timestep < 0:
        return None
    if not isinstance(length, int) or length <= 0:
        return None
    if length != int(expected_length):
        return None

    start_ts = pd.to_datetime(start)
    if timestep == 0:
        return pd.DatetimeIndex([start_ts] * length)
    return pd.date_range(start=start_ts, periods=length, freq=pd.to_timedelta(timestep, unit="s"))


def _time_index_from_config(config: Optional[Dict[str, Any]], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """
    config["simulation"]["index"] が dict（start/end/timestep/length）なら DatetimeIndex を復元する。
    expected_length と length が一致しない場合は None。
    """
    if not isinstance(config, dict):
        return None
    sim = config.get("simulation")
    if not isinstance(sim, dict):
        return None
    spec = sim.get("index")
    if not isinstance(spec, dict):
        return None
    return _time_index_from_spec(spec, expected_length=expected_length)


def _time_index_from_output(resp_json: Dict[str, Any], *, expected_length: int) -> Optional[pd.DatetimeIndex]:
    """
    APIレスポンス（/runのJSON）に含まれる output.index から DatetimeIndex を復元する。
    expected_length と length が一致しない場合は None。
    """
    output = _output_block(resp_json)
    spec = output.get("index")
    if not isinstance(spec, dict):
        return None
    return _time_index_from_spec(spec, expected_length=expected_length)


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
    "_time_index_from_config",
    "_time_index_from_output",
]



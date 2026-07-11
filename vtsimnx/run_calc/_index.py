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
]

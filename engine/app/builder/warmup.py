"""典型年の末尾を先頭に付け、建物を評価開始より前から積分する。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)


def _parse_month_day(value: str) -> tuple[int, int]:
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    parts = text.replace("/", "-").split("-")
    if len(parts) == 3:
        month, day = int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        month, day = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"simulation.warmup.start は MM-DD です。got {value!r}")
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"simulation.warmup.start の月日が不正です: {value!r}")
    return month, day


def _index_from_sim(sim: dict[str, Any]) -> pd.DatetimeIndex:
    spec = sim.get("index")
    if not isinstance(spec, dict):
        raise ValueError("simulation.warmup には simulation.index が必要です。")
    start = spec.get("start")
    timestep = spec.get("timestep")
    length = spec.get("length")
    if not isinstance(start, str) or not start:
        raise ValueError("simulation.warmup には index.start が必要です。")
    if not isinstance(timestep, (int, float)) or int(timestep) <= 0:
        raise ValueError("simulation.warmup には正の index.timestep が必要です。")
    if not isinstance(length, int) or length < 2:
        raise ValueError("simulation.warmup には index.length >= 2 が必要です。")
    start_ts = pd.to_datetime(start)
    return pd.date_range(start=start_ts, periods=length, freq=pd.to_timedelta(int(timestep), unit="s"))


def _fmt(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _prepend_value(value: Any, head: list[int], orig_len: int) -> Any:
    if isinstance(value, dict):
        return {key: _prepend_value(item, head, orig_len) for key, item in value.items()}
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == orig_len and (not value or not isinstance(value[0], (dict, list))):
            return [value[i] for i in head] + value
        return [_prepend_value(item, head, orig_len) for item in value]
    return value


def apply_typical_year_warmup(raw: dict[str, Any]) -> None:
    """raw をその場で伸ばす。典型年の warmup.start 以降を前年として先頭に付ける。"""
    sim = raw.get("simulation")
    if not isinstance(sim, dict):
        return
    warmup = sim.get("warmup")
    if warmup in (None, False):
        return
    if isinstance(warmup, str):
        warmup = {"start": warmup}
        sim["warmup"] = warmup
    if not isinstance(warmup, dict):
        raise TypeError("simulation.warmup は dict か開始日の文字列です。")
    if warmup.get("steps"):
        return

    start_spec = warmup.get("start")
    if not start_spec:
        raise ValueError("simulation.warmup.start を指定してください（例: '12-15'）。")

    idx = _index_from_sim(sim)
    month, day = _parse_month_day(str(start_spec))
    first = idx[0]
    if (int(first.month), int(first.day)) >= (month, day):
        raise ValueError(
            "simulation.warmup.start は評価開始より後の典型年日付にしてください。"
            f" index.start={_fmt(first)}, warmup.start={month:02d}-{day:02d}"
        )

    head_mask = (idx.month > month) | ((idx.month == month) & (idx.day >= day))
    head = [int(i) for i, flag in enumerate(head_mask) if bool(flag)]
    if not head:
        raise ValueError(
            f"simulation.warmup.start={month:02d}-{day:02d} が index に含まれません。"
        )

    orig_len = int(len(idx))
    year0 = int(first.year)
    shifted: list[pd.Timestamp] = []
    for ts in idx[head]:
        ts = pd.Timestamp(ts)
        try:
            shifted.append(ts.replace(year=year0 - 1))
        except ValueError:
            shifted.append(ts.replace(year=year0 - 1, day=28))
    warmup_index = pd.DatetimeIndex(shifted)
    new_index = pd.DatetimeIndex(warmup_index.tolist() + idx.tolist())
    spec = sim.get("index")
    if isinstance(spec, dict) and spec.get("timestep"):
        timestep = int(spec["timestep"])
    elif len(idx) >= 2:
        timestep = int(round((idx[1] - idx[0]).total_seconds()))
    else:
        raise ValueError("simulation.warmup には正の index.timestep が必要です。")

    for key, value in list(raw.items()):
        if key == "simulation":
            continue
        raw[key] = _prepend_value(value, head, orig_len)

    warmup["start"] = f"{month:02d}-{day:02d}"
    warmup["steps"] = int(len(head))
    warmup["eval_start"] = _fmt(idx[0])
    warmup["eval_length"] = orig_len
    sim["index"] = {
        "start": _fmt(new_index[0]),
        "end": _fmt(new_index[-1]),
        "timestep": timestep,
        "length": int(len(new_index)),
    }
    logger.info(
        "助走を適用しました: start=%s steps=%d eval_length=%d total_length=%d",
        warmup["start"],
        warmup["steps"],
        orig_len,
        int(len(new_index)),
    )


def warmup_output_skip(sim: dict[str, Any] | None) -> int:
    """結果系列の先頭から落とす助走ステップ数。"""
    if not isinstance(sim, dict):
        return 0
    warmup = sim.get("warmup")
    if not isinstance(warmup, dict):
        return 0
    steps = warmup.get("steps")
    if isinstance(steps, int) and steps > 0:
        return steps
    return 0

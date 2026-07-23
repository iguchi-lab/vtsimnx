"""Parse solver.log for convergence / iteration / DirectT cache metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_RE_THERMAL = re.compile(
    r"熱計算\(線形\):\s*(収束|未収束).*?RMSE=([0-9eE.+-]+).*?maxBalance=([0-9eE.+-]+)"
)
# 新語彙「物理収支合格」と旧「圧力計算収束」/ Fallback 残差行の両方を拾う
_RE_PRESSURE = re.compile(
    r"(?:圧力計算収束|物理収支合格|\[Fallback\]\s*(?:収束|物理収支合格))"
    r".*?(?:residual|mass_maxAbs)=([0-9eE.+-]+)"
)
_RE_PRESSURE_FAIL = re.compile(
    r"(?:圧力計算未収束|物理収支未達|\[Fallback\]\s*(?:未収束|物理収支未達))"
)
_RE_COUPLED_ITERS = re.compile(r"総連成反復回数:\s*(\d+)")
_RE_AIRCON_RECOMPUTE = re.compile(r"再計算を実行します")
_RE_AIRCON_LOOP_OK = re.compile(r"エアコン制御ループ\s+\d+\s+が収束しました")
_RE_DIRECTT_STATS = re.compile(
    r"DirectT cache stats:.*?topoRebuild=(\d+).*?patternRebuild=(\d+).*?luFactorize=(\d+)"
)
# `[INFO]` に含まれる Inf を誤検出しない（単語境界で Inf/NaN のみ）
_RE_NAN = re.compile(
    r"(?<![A-Za-z])(?:NaN|\+?-?Inf(?:inity)?|not finite)(?![A-Za-z])",
    re.IGNORECASE,
)


@dataclass
class SolverLogMetrics:
    thermal_converged: list[bool] = field(default_factory=list)
    thermal_rmse: list[float] = field(default_factory=list)
    thermal_max_balance: list[float] = field(default_factory=list)
    pressure_residuals: list[float] = field(default_factory=list)
    pressure_failed: bool = False
    coupled_iterations: list[int] = field(default_factory=list)
    aircon_recompute_count: int = 0
    aircon_loop_converged_count: int = 0
    lu_factorize: int | None = None
    topo_rebuild: int | None = None
    pattern_rebuild: int | None = None
    nan_inf_mentions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermal_converged_all": bool(self.thermal_converged) and all(self.thermal_converged),
            "thermal_max_balance_peak": max(self.thermal_max_balance) if self.thermal_max_balance else None,
            "thermal_rmse_peak": max(self.thermal_rmse) if self.thermal_rmse else None,
            "pressure_residual_peak": max(self.pressure_residuals) if self.pressure_residuals else None,
            "pressure_failed": self.pressure_failed,
            "coupled_iterations_max": max(self.coupled_iterations) if self.coupled_iterations else None,
            "coupled_iterations_sum": sum(self.coupled_iterations) if self.coupled_iterations else 0,
            "aircon_recompute_count": self.aircon_recompute_count,
            "aircon_loop_converged_count": self.aircon_loop_converged_count,
            "lu_factorize": self.lu_factorize,
            "topo_rebuild": self.topo_rebuild,
            "pattern_rebuild": self.pattern_rebuild,
            "nan_inf_mentions": self.nan_inf_mentions,
        }


def parse_solver_log(text: str) -> SolverLogMetrics:
    m = SolverLogMetrics()
    for match in _RE_THERMAL.finditer(text):
        m.thermal_converged.append(match.group(1) == "収束")
        m.thermal_rmse.append(float(match.group(2)))
        m.thermal_max_balance.append(float(match.group(3)))
    for match in _RE_PRESSURE.finditer(text):
        m.pressure_residuals.append(float(match.group(1)))
    if _RE_PRESSURE_FAIL.search(text):
        m.pressure_failed = True
    for match in _RE_COUPLED_ITERS.finditer(text):
        m.coupled_iterations.append(int(match.group(1)))
    m.aircon_recompute_count = len(_RE_AIRCON_RECOMPUTE.findall(text))
    m.aircon_loop_converged_count = len(_RE_AIRCON_LOOP_OK.findall(text))
    # Use the last DirectT stats line (cumulative counters).
    for match in _RE_DIRECTT_STATS.finditer(text):
        m.topo_rebuild = int(match.group(1))
        m.pattern_rebuild = int(match.group(2))
        m.lu_factorize = int(match.group(3))
    m.nan_inf_mentions = len(_RE_NAN.findall(text))
    return m

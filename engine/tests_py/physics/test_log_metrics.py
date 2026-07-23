"""Unit tests for solver.log metric parsing (no solver binary required)."""

from __future__ import annotations

import pytest

from .log_metrics import parse_solver_log


@pytest.mark.physics
def test_info_prefix_is_not_treated_as_inf():
    text = (
        "[熱] [INFO] DirectT: LU backend=KLU\n"
        "[圧力] [INFO] 圧力ゲージ固定: soft anchor 数=1\n"
        "[熱] 熱計算(線形): 収束 (method=KLU, RMSE=1.0e-12, maxBalance=2.0e-12, time=0.1ms)\n"
        "[連成] 圧力-温度連成計算-エアコン制御ループ 1 が収束しました。\n"
        "[ts=1] [連成] タイムステップ終了  総連成反復回数: 2\n"
    )
    m = parse_solver_log(text)
    assert m.nan_inf_mentions == 0
    assert m.thermal_converged == [True]
    assert m.aircon_loop_converged_count == 1
    assert m.coupled_iterations == [2]


@pytest.mark.physics
def test_real_inf_and_nan_are_still_detected():
    assert parse_solver_log("value became Inf").nan_inf_mentions == 1
    assert parse_solver_log("NaN in residual").nan_inf_mentions == 1
    assert parse_solver_log("not finite temperature").nan_inf_mentions == 1


@pytest.mark.physics
def test_pressure_physical_balance_vocabulary():
    ok = parse_solver_log("[圧力] 物理収支合格 | mass_maxAbs=1.23e-08 | tol=1e-06")
    assert ok.pressure_residuals == [1.23e-08]
    assert not ok.pressure_failed

    bad = parse_solver_log("[圧力] [WARN] プライマリは物理収支未達のためフォールバックへ移行")
    assert bad.pressure_failed

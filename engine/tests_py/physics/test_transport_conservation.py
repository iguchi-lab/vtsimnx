"""Independent transport invariants from the November 2026 audit."""

import math

import pytest

from .conftest import requires_solver
from .helpers import read_series, run_from_raw, read_solver_log

pytestmark = [pytest.mark.physics, requires_solver]


def config(length=3, step=3600):
    return {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00",
                "end": "2025-01-01T02:00:00",
                "length": length,
                "timestep": step,
            }
        },
        "nodes": [],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


def solve(raw, workdir, run_id):
    return run_from_raw(
        raw_config=raw,
        run_id=run_id,
        tmp_base_dir=workdir,
        add_surface=False,
        add_capacity=False,
    )


@pytest.mark.parametrize("mode", ["sensible", "total"])
def test_hrv_preserves_uniform_nonzero_concentration(mode, solver_workdir):
    raw = config()
    raw["nodes"] = [
        {"key": "OA", "t": 20, "c": 400},
        {"key": "R", "t": 20, "v": 50, "c": 400, "calc_c": True},
    ]
    raw["heat_recovery_vent"] = [
        {"key": "HRV", "outdoor": "OA", "room": "R", "vol": 50 / 3600, "recovery": mode}
    ]
    out, art = solve(raw, solver_workdir, "hrv_c_" + mode)
    assert read_series(art, out, "concentration_c", "R") == pytest.approx([400] * 3)


def test_hrv_transmits_current_outdoor_concentration_without_step_delay(solver_workdir):
    raw = config()
    raw["nodes"] = [
        {"key": "OA", "t": 20, "c": [400, 600, 800]},
        {"key": "R", "t": 20, "v": 50, "c": 400, "calc_c": True},
    ]
    raw["heat_recovery_vent"] = [
        {"key": "HRV", "outdoor": "OA", "room": "R", "vol": 50 / 3600}
    ]
    out, art = solve(raw, solver_workdir, "hrv_c_step")
    # 1 ACHの陰解法、HRVは容量なしの同時混合。
    assert read_series(art, out, "concentration_c", "R") == pytest.approx(
        [400, 500, 650]
    )


def test_hrv_humidity_flux_uses_same_state_as_humidity_solver(solver_workdir):
    raw = config()
    raw["nodes"] = [
        {"key": "OA", "t": 20, "x": 0.01},
        {"key": "R", "t": 20, "v": 50, "x": 0.01, "calc_x": True},
    ]
    raw["heat_recovery_vent"] = [
        {
            "key": "HRV",
            "outdoor": "OA",
            "room": "R",
            "vol": 50 / 3600,
            "recovery": "total",
        }
    ]
    out, art = solve(raw, solver_workdir, "hrv_x_flux")
    assert read_series(art, out, "humidity_x", "R") == pytest.approx([0.01] * 3)
    for edge in ("OA->HRV", "HRV->R", "R->HRV_exhaust", "HRV_exhaust->OA"):
        assert read_series(art, out, "humidity_flux", edge) == pytest.approx(
            [1.2 * (50 / 3600) * 0.01] * 3
        )


@pytest.mark.parametrize("step", [3600, 600, 60, 10])
def test_closed_two_room_concentration_conserves_total_and_refines(
    step, solver_workdir
):
    raw = config(length=3600 // step, step=step)
    raw["nodes"] = [
        {"key": "A", "t": 20, "v": 50, "c": 1000, "calc_c": True},
        {"key": "B", "t": 20, "v": 100, "c": 0, "calc_c": True},
    ]
    raw["ventilation_branches"] = [
        {"key": "A->B", "vol": 50 / 3600},
        {"key": "B->A", "vol": 50 / 3600},
    ]
    out, art = solve(raw, solver_workdir, "closed_c_" + str(step))
    a = read_series(art, out, "concentration_c", "A")
    b = read_series(art, out, "concentration_c", "B")
    for ca, cb in zip(a, b):
        assert ca >= 0 and cb >= 0
        assert 50 * ca + 100 * cb == pytest.approx(50000, rel=1e-7)
    # 連立陰解法の独立解。刻みを小さくすると厳密解へ近づく。
    decay = (1 + 1.5 * step / 3600) ** (-3600 // step)
    assert a[-1] == pytest.approx(1000 / 3 + 2000 / 3 * decay, rel=1e-6)
    exact = 1000 / 3 + 2000 / 3 * math.exp(-1.5)
    assert abs(a[-1] - exact) <= abs(600 - exact) + 1e-5
    if step == 10:
        assert abs(a[-1] - exact) / exact < 0.001


def test_zero_volume_filter_junction_keeps_removal_efficiency(solver_workdir):
    raw = config(length=1)
    raw["nodes"] = [
        {"key": "OA", "t": 20, "c": 1000},
        {"key": "J", "t": 20, "v": 0, "calc_c": True},
        {"key": "R", "t": 20, "v": 3600, "c": 0, "calc_c": True},
    ]
    raw["ventilation_branches"] = [
        {"key": "OA->J", "vol": 1, "eta": 0.9},
        {"key": "J->R", "vol": 1},
        {"key": "R->OA", "vol": 1},
    ]
    out, art = solve(raw, solver_workdir, "zero_c_filter")
    assert read_series(art, out, "concentration_c", "J") == pytest.approx([100])
    assert read_series(art, out, "concentration_c", "R") == pytest.approx([50])


def test_iteration_settings_reach_solver_and_change_limit(solver_workdir):
    raw = config()
    raw["nodes"] = [{"key": "R", "t": 20, "v": 50, "x": 0.01, "calc_x": True}]
    raw["simulation"]["iteration"] = {"max_inner": 7, "max_aircon_control": 9}
    raw["simulation"]["coupling"] = {"moisture_enabled": False}
    out, art = solve(raw, solver_workdir, "input_controls")
    log = read_solver_log(art, out)
    assert "最大内部反復回数（設定値）: 7" in log
    assert "最大連成反復回数: 7, 最大空調制御反復回数: 9" in log
    assert "3ネットワーク連成: false" in log


def test_http_run_keeps_hrv_and_controls_all_the_way_to_solver(
    solver_workdir, monkeypatch
):
    from fastapi.testclient import TestClient
    from app.main import app
    import app.solver_runner as sr
    import app.services.simulation as svc

    for key in ("VTSIMNX_API_KEY", "VTSIMNX_API_KEYS", "VTSIMNX_API_KEYS_JSON"):
        monkeypatch.delenv(key, raising=False)
    # APIの他テストに残ったモックではなく実ソルバを使う。
    monkeypatch.setattr(svc, "run_solver", sr.run_solver)
    raw = config()
    raw["nodes"] = [
        {"key": "OA", "t": 20, "c": 400},
        {"key": "R", "t": 20, "v": 50, "c": 400, "calc_c": True},
    ]
    raw["heat_recovery_vent"] = [
        {"key": "HRV", "outdoor": "OA", "room": "R", "vol": 50 / 3600}
    ]
    raw["simulation"].update({"iteration": {"max_inner": 7}, "log": {"verbosity": 1}})
    with TestClient(app) as client:
        response = client.post("/run", json={"config": raw, "unknown_keys": "error"})
    assert response.status_code == 200, response.json()
    payload = response.json()
    assert payload["warnings"] == []
    out = payload["result"]
    assert out["status"] == "ok"
    art = sr.resolve_artifact_path(out["artifact_dir"])
    assert art is not None
    assert read_series(art, out, "concentration_c", "R") == pytest.approx([400] * 3)
    # 機器が削除されて無換気でも400は維持するため、枝の存在も独立に確認。
    assert read_series(art, out, "vent_flow_rate", "OA->HRV") == pytest.approx(
        [50 / 3600] * 3
    )
    assert "最大内部反復回数（設定値）: 7" in read_solver_log(art, out)

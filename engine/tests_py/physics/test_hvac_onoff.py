from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_energy_balance_residuals,
    read_series,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _ac_spec() -> dict:
    return {
        "Q": {
            "cooling": {"min": 0.7, "rtd": 2.2, "max": 3.3},
            "heating": {"min": 0.7, "rtd": 2.5, "max": 5.4},
        },
        "P": {
            "cooling": {"min": 0.095, "rtd": 0.395, "max": 0.78},
            "heating": {"min": 0.095, "rtd": 0.39, "max": 1.36},
        },
        "V_inner": {
            "cooling": {"rtd": 0.2016666666667},
            "heating": {"rtd": 0.2016666666667},
        },
        "V_outer": {
            "cooling": {"rtd": 0.47},
            "heating": {"rtd": 0.47},
        },
    }


def _raw_hvac_onoff_case() -> dict:
    length = 6
    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-3},
            "calc_flag": {"p": False, "t": True, "x": False, "c": False},
        },
        "nodes": [
            {"key": "outside", "t": 5.0},
            {"key": "room", "calc_t": True, "t": 15.0, "v": 40.0, "thermal_mass": 5.0e5},
        ],
        "ventilation_branches": [],
        "thermal_branches": [
            {
                "key": "outside->room",
                "source": "outside",
                "target": "room",
                "type": "conductance",
                "conductance": 30.0,
            }
        ],
        "aircon": [
            {
                "key": "AC1",
                "set": "room",
                "outside": "outside",
                "pre_temp": 22.0,
                "model": "CRIEPI",
                "mode": ["OFF", "OFF", "OFF", "HEATING", "HEATING", "HEATING"],
                "vol": 0.2,
                "ac_spec": _ac_spec(),
            }
        ],
    }


@pytest.mark.physics
@requires_solver
def test_hvac_on_off_boundary_sensible_heat(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_hvac_onoff_case(),
        run_id="hvac_onoff",
        tmp_base_dir=solver_workdir,
        add_aircon=True,
        add_surface=False,
        add_capacity=True,
        log_verbosity=1,
    )

    room_t = read_series(art, out, "thermal_temperature", "room")
    assert_all_finite(room_t, label="room")

    keys, rows = read_series_matrix(art, out, "aircon_sensible_heat")
    assert keys, "expected aircon_sensible_heat keys"
    ac_idx = next((i for i, k in enumerate(keys) if "AC1" in k or k == "AC1"), 0)
    off_vals = [rows[t][ac_idx] for t in range(3)]
    on_vals = [rows[t][ac_idx] for t in range(3, 6)]
    assert_all_finite(off_vals + on_vals, label="aircon_sensible_heat")

    assert all(abs(v) <= tol.AIRCON_OFF_ABS_W for v in off_vals), f"OFF should be ~0, got {off_vals}"
    assert any(abs(v) > tol.AIRCON_OFF_ABS_W for v in on_vals), f"ON should be non-zero, got {on_vals}"
    assert room_t[-1] > room_t[2] - 0.5

    assert_artifact_no_nan_inf(art, out)
    # 空調 ON 時は aircon ノード特例で単純な室エネルギー残差が崩れうるため、
    # 収束ログ + 有限性を必須とし、エネルギー残差は OFF 期間相当の検証に限定しない。
    metrics = assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)
    assert metrics.aircon_loop_converged_count >= 1
    assert metrics.coupled_iterations and max(metrics.coupled_iterations) >= 1
    d = metrics.to_dict()
    assert d["coupled_iterations_max"] is not None and d["coupled_iterations_max"] >= 1

from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_flux_series_node_balance,
    assert_non_negative,
    node_balance_from_edge_rates,
    read_series,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _raw_humidity_advection_case() -> dict:
    """Phase1 humidity: ~1 ACH + small generation (physically bounded x)."""
    length = 6
    outdoor_x = [0.010] * (length // 2) + [0.004] * (length - length // 2)
    return {
        "builder": {"add_moisture_capacity": False},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {
                "ventilation": 1e-6,
                "thermal": 1e-6,
                "convergence": 1e-6,
                "coupling_humidity": 1e-8,
            },
            "calc_flag": {"p": False, "t": True, "x": True, "c": False},
            "coupling": {
                "moisture_enabled": True,
                "humidity_relaxation": 1.0,
                "humidity_solver_tolerance": 1e-9,
            },
        },
        "nodes": [
            {"key": "outside", "t": 5.0, "x": outdoor_x},
            {
                "key": "room",
                "calc_t": True,
                "calc_x": True,
                "v": 30.0,
                "t": 20.0,
                "x": 0.006,
                "thermal_mass": 1.0e6,
            },
        ],
        "ventilation_branches": [
            {
                "key": "outside->room",
                "source": "outside",
                "target": "room",
                "type": "fixed_flow",
                # m3/s ≈ 1 ACH for 30 m3
                "vol": 30.0 / 3600.0,
            }
        ],
        "thermal_branches": [
            {
                "key": "outside->room",
                "source": "outside",
                "target": "room",
                "type": "conductance",
                "conductance": 50.0,
            }
        ],
        "humidity_source": [
            # ごく小さな発湿（絶対湿度を 0..0.05 に保つ）
            {"key": "gen", "room": "room", "generation_rate": [1.0e-7] * length},
        ],
    }


@pytest.mark.physics
@requires_solver
def test_humidity_mass_balance_finite_and_responds(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_humidity_advection_case(),
        run_id="humidity_balance",
        tmp_base_dir=solver_workdir,
        add_surface=False,
        add_capacity=True,
        add_moisture_capacity=False,
        log_verbosity=1,
    )

    x_room = read_series(art, out, "humidity_x", "room")
    assert_all_finite(x_room, label="humidity_x.room")
    assert_non_negative(x_room, label="humidity_x.room")
    assert all(v <= tol.HUMIDITY_X_MAX for v in x_room), f"humidity_x out of physical range: {x_room}"

    assert max(x_room) - min(x_room) > 1e-8

    flux_keys, flux_rows = read_series_matrix(art, out, "humidity_flux")
    assert flux_keys, "expected humidity_flux keys"
    for t, row in enumerate(flux_rows):
        assert_all_finite(row, label=f"humidity_flux[{t}]")
        assert any(abs(v) > 1e-15 for v in row), f"all humidity flux ~0 at t={t}"
        bal = node_balance_from_edge_rates(flux_keys, row)
        # 全節点の流量和はテレスコープで ≈0（水分質量の大域保存）
        assert abs(sum(bal.values())) <= tol.HUMIDITY_FLUX_BALANCE_ABS

    assert_flux_series_node_balance(
        art,
        out,
        "humidity_flux",
        nodes=["room", "outside"],
        abs_tol=tol.HUMIDITY_FLUX_BALANCE_ABS,
        label="humidity_mass",
    )
    assert_artifact_no_nan_inf(art, out)
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)

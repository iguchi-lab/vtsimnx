from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_energy_balance_residuals,
    read_series,
    read_solver_log,
    run_from_raw,
)


def _raw_extreme_params_case(*, thermal_mass: float, u_value: float) -> dict:
    length = 4
    return {
        "builder": {"surface_layer_method": "rc"},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T03:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "thermal_mass": thermal_mass},
            {"key": "outside", "t": [0.0] * length, "calc_t": False},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "room->outside",
                "part": "wall",
                "area": 10.0,
                "u_value": u_value,
                "alpha_i": 4.4,
                "alpha_o": 23.0,
            }
        ],
    }


@pytest.mark.physics
@requires_solver
def test_near_zero_capacity_remains_finite(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_extreme_params_case(thermal_mass=1.0, u_value=0.5),
        run_id="near_zero_capa",
        tmp_base_dir=solver_workdir,
        add_surface=True,
        add_capacity=True,
        add_surface_solar=False,
        add_surface_nocturnal=False,
        add_surface_radiation=False,
        log_verbosity=1,
    )
    room_t = read_series(art, out, "thermal_temperature", "room")
    assert_all_finite(room_t, label="room")
    assert room_t[-1] < room_t[0]
    assert_artifact_no_nan_inf(art, out)
    assert_energy_balance_residuals(art, out, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)


@pytest.mark.physics
@requires_solver
def test_extreme_uvalue_remains_finite(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_extreme_params_case(thermal_mass=2.0e6, u_value=50.0),
        run_id="extreme_u",
        tmp_base_dir=solver_workdir,
        add_surface=True,
        add_capacity=True,
        add_surface_solar=False,
        add_surface_nocturnal=False,
        add_surface_radiation=False,
        log_verbosity=1,
    )
    room_t = read_series(art, out, "thermal_temperature", "room")
    assert_all_finite(room_t, label="room")
    assert room_t[-1] < room_t[0]
    assert_artifact_no_nan_inf(art, out)
    assert_energy_balance_residuals(art, out, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)

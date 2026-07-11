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
from . import tolerances as tol


def _raw_cooling_case(*, length: int, timestep: int) -> dict:
    return {
        "builder": {"surface_layer_method": "rc"},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": timestep,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "thermal_mass": 2.0e6},
            {"key": "outside", "t": [0.0] * length, "calc_t": False},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "room->outside",
                "part": "wall",
                "area": 10.0,
                "u_value": 0.5,
                "alpha_i": 4.4,
                "alpha_o": 23.0,
            }
        ],
    }


@pytest.mark.physics
@requires_solver
def test_timestep_refinement_converges_endpoint(solver_workdir):
    out_coarse, art_coarse = run_from_raw(
        raw_config=_raw_cooling_case(length=6, timestep=3600),
        run_id="dt_coarse",
        tmp_base_dir=solver_workdir,
        add_surface=True,
        add_capacity=True,
        add_surface_solar=False,
        add_surface_nocturnal=False,
        add_surface_radiation=False,
        log_verbosity=1,
    )
    out_fine, art_fine = run_from_raw(
        raw_config=_raw_cooling_case(length=12, timestep=1800),
        run_id="dt_fine",
        tmp_base_dir=solver_workdir,
        add_surface=True,
        add_capacity=True,
        add_surface_solar=False,
        add_surface_nocturnal=False,
        add_surface_radiation=False,
        log_verbosity=1,
    )

    coarse = read_series(art_coarse, out_coarse, "thermal_temperature", "room")
    fine = read_series(art_fine, out_fine, "thermal_temperature", "room")
    assert_all_finite(coarse, label="coarse")
    assert_all_finite(fine, label="fine")

    assert abs(coarse[-1] - fine[-1]) <= tol.TIMESTEP_ENDPOINT_ABS_K
    assert fine[-1] < fine[0]
    assert coarse[-1] < coarse[0]

    assert_artifact_no_nan_inf(art_coarse, out_coarse)
    assert_artifact_no_nan_inf(art_fine, out_fine)
    assert_energy_balance_residuals(art_coarse, out_coarse, nodes=["room"])
    assert_energy_balance_residuals(art_fine, out_fine, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art_coarse, out_coarse), expect_thermal=True)
    assert_convergence_from_log(read_solver_log(art_fine, out_fine), expect_thermal=True)

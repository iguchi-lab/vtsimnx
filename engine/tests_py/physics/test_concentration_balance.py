"""Minimal contaminant (concentration) mass-balance baseline."""

from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_non_negative,
    node_balance_from_edge_rates,
    read_series,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _raw_concentration_case() -> dict:
    length = 4
    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T03:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": True, "x": False, "c": True},
        },
        "nodes": [
            {"key": "outside", "t": 10.0, "c": [0.0] * length},
            {
                "key": "room",
                "calc_t": True,
                "calc_c": True,
                "v": 30.0,
                "t": 20.0,
                "c": 1.0e-6,
                "thermal_mass": 1.0e6,
            },
        ],
        "ventilation_branches": [
            {
                "key": "outside->room",
                "type": "fixed_flow",
                "vol": 30.0 / 3600.0,
            }
        ],
        "thermal_branches": [
            {
                "key": "outside->room",
                "type": "conductance",
                "conductance": 20.0,
            }
        ],
    }


@pytest.mark.physics
@requires_solver
def test_concentration_mass_balance_nonnegative(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_concentration_case(),
        run_id="concentration_balance",
        tmp_base_dir=solver_workdir,
        add_surface=False,
        add_capacity=True,
        log_verbosity=1,
    )

    c_room = read_series(art, out, "concentration_c", "room")
    assert_all_finite(c_room, label="concentration_c.room")
    assert_non_negative(c_room, label="concentration_c.room", tol=1e-15)
    assert all(v >= tol.CONCENTRATION_MIN for v in c_room)

    flux_keys, flux_rows = read_series_matrix(art, out, "concentration_flux")
    if flux_keys:
        for t, row in enumerate(flux_rows):
            assert_all_finite(row, label=f"concentration_flux[{t}]")
            bal = node_balance_from_edge_rates(flux_keys, row)
            assert abs(sum(bal.values())) <= tol.CONCENTRATION_FLUX_BALANCE_ABS

    assert_artifact_no_nan_inf(art, out)
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)

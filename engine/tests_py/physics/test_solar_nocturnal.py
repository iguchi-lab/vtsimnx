from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    assert_energy_balance_residuals,
    assert_non_negative,
    read_series,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _raw_solar_nocturnal_case() -> dict:
    length = 4
    return {
        "builder": {
            "surface_layer_method": "rc",
            "add_surface_solar": True,
            "add_surface_nocturnal": True,
            "add_surface_radiation": False,
        },
        "simulation": {
            "index": {
                "start": "2000-07-01T12:00:00",
                "end": "2000-07-01T15:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "thermal_mass": 2.0e6},
            {"key": "outside", "t": [25.0] * length, "calc_t": False},
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
                "epsilon": 0.9,
                "solar": [400.0, 400.0, 0.0, 0.0],
                "nocturnal": [0.0, 0.0, 50.0, 50.0],
            }
        ],
    }


@pytest.mark.physics
@requires_solver
def test_solar_and_nocturnal_heat_rates_respond(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_solar_nocturnal_case(),
        run_id="solar_nocturnal",
        tmp_base_dir=solver_workdir,
        add_surface=True,
        add_capacity=True,
        add_surface_solar=True,
        add_surface_nocturnal=True,
        add_surface_radiation=False,
        log_verbosity=1,
    )

    room_t = read_series(art, out, "thermal_temperature", "room")
    assert_all_finite(room_t, label="room")

    solar_keys, solar_rows = read_series_matrix(art, out, "thermal_heat_rate_solar_gain")
    noct_keys, noct_rows = read_series_matrix(art, out, "thermal_heat_rate_nocturnal_loss")
    assert solar_keys, "expected solar_gain branches"
    assert noct_keys, "expected nocturnal_loss branches"

    solar_early = sum(abs(v) for v in solar_rows[0])
    solar_late = sum(abs(v) for v in solar_rows[-1])
    noct_early = sum(abs(v) for v in noct_rows[0])
    noct_late = sum(abs(v) for v in noct_rows[-1])
    assert solar_early > 1e-6
    assert solar_late < solar_early
    assert noct_late > 1e-6
    assert noct_early < noct_late

    # 日射ゲインは非負（入力 solar>=0 のとき）
    for t, row in enumerate(solar_rows):
        assert_all_finite(row, label=f"solar_gain[{t}]")
        assert_non_negative(row, label=f"solar_gain[{t}]", tol=1e-9)
    for t, row in enumerate(noct_rows):
        assert_all_finite(row, label=f"nocturnal_loss[{t}]")

    assert_artifact_no_nan_inf(art, out)
    assert_energy_balance_residuals(art, out, nodes=["room"])
    assert_convergence_from_log(read_solver_log(art, out), expect_thermal=True)

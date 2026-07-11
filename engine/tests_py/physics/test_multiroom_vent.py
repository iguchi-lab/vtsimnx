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


def _raw_two_room_opening_case() -> dict:
    length = 6
    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T05:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": True, "t": True, "x": False, "c": False},
        },
        "nodes": [
            {"key": "out_h", "t": [10.0] * length, "p": [20.0] * length, "calc_t": False, "calc_p": False},
            {"key": "out_l", "t": [10.0] * length, "p": [0.0] * length, "calc_t": False, "calc_p": False},
            {"key": "room_a", "t": 25.0, "calc_t": True, "calc_p": True, "v": 50.0, "thermal_mass": 1.0e6},
            {"key": "room_b", "t": 15.0, "calc_t": True, "calc_p": True, "v": 50.0, "thermal_mass": 1.0e6},
        ],
        "ventilation_branches": [
            {"key": "out_h->room_a", "type": "simple_opening", "alpha": 0.7, "area": 0.5},
            {"key": "room_a->room_b", "type": "simple_opening", "alpha": 0.7, "area": 1.0},
            {"key": "room_b->out_l", "type": "simple_opening", "alpha": 0.7, "area": 0.5},
        ],
        "thermal_branches": [
            {"key": "room_a->room_b", "type": "conductance", "conductance": 10.0},
        ],
    }


@pytest.mark.physics
@requires_solver
def test_multiroom_vent_coupling_temperatures_approach(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_two_room_opening_case(),
        run_id="multiroom_vent",
        tmp_base_dir=solver_workdir,
        add_surface=False,
        add_capacity=True,
        log_verbosity=1,
    )

    ta = read_series(art, out, "thermal_temperature", "room_a")
    tb = read_series(art, out, "thermal_temperature", "room_b")
    assert_all_finite(ta, label="room_a")
    assert_all_finite(tb, label="room_b")
    assert abs(ta[0] - tb[0]) > abs(ta[-1] - tb[-1]) - 1e-3

    flow_keys, flow_rows = read_series_matrix(art, out, "vent_flow_rate")
    assert set(flow_keys) >= {"out_h->room_a", "room_a->room_b", "room_b->out_l"}
    for t, row in enumerate(flow_rows):
        assert_all_finite(row, label=f"vent_flow_rate[{t}]")
        d = dict(zip(flow_keys, row))
        assert abs(d["out_h->room_a"]) > 1e-6
        # 空気質量収支残差（節点）
        bal_a = d["out_h->room_a"] - d["room_a->room_b"]
        bal_b = d["room_a->room_b"] - d["room_b->out_l"]
        assert abs(bal_a) <= tol.VENT_MASS_BALANCE_ABS, f"room_a mass imbalance at t={t}: {bal_a}"
        assert abs(bal_b) <= tol.VENT_MASS_BALANCE_ABS, f"room_b mass imbalance at t={t}: {bal_b}"

    _press_keys, press_rows = read_series_matrix(art, out, "vent_pressure")
    for t, row in enumerate(press_rows):
        assert_all_finite(row, label=f"vent_pressure[{t}]")

    assert_artifact_no_nan_inf(art, out)
    assert_energy_balance_residuals(art, out, nodes=["room_a", "room_b"])
    assert_convergence_from_log(
        read_solver_log(art, out),
        expect_thermal=True,
        expect_pressure=True,
    )

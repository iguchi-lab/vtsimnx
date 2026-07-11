from __future__ import annotations

import pytest

from .conftest import requires_solver
from .helpers import (
    assert_all_finite,
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    read_series_matrix,
    read_solver_log,
    run_from_raw,
)
from . import tolerances as tol


def _raw_stiff_pressure_network() -> dict:
    length = 3
    openings = []
    for i in range(8):
        openings.append(
            {
                "key": f"out_h->room_a||oa{i}",
                "type": "simple_opening",
                "alpha": 0.01 if i % 2 == 0 else 0.99,
                "area": 1e-4 if i % 2 == 0 else 2.0,
            }
        )
        openings.append(
            {
                "key": f"room_a->room_b||ab{i}",
                "type": "simple_opening",
                "alpha": 0.05 if i % 3 == 0 else 0.8,
                "area": 1e-3 if i % 3 == 0 else 1.5,
            }
        )
        openings.append(
            {
                "key": f"room_b->out_l||bo{i}",
                "type": "simple_opening",
                "alpha": 0.02 if i % 2 else 0.95,
                "area": 5e-4 if i % 2 else 1.8,
            }
        )

    return {
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T02:00:00",
                "timestep": 3600,
                "length": length,
            },
            "tolerance": {"ventilation": 1e-5, "thermal": 1e-5, "convergence": 1e-5},
            "calc_flag": {"p": True, "t": False, "x": False, "c": False},
        },
        "nodes": [
            {"key": "out_h", "p": [30.0] * length, "t": 10.0, "calc_p": False},
            {"key": "out_l", "p": [0.0] * length, "t": 10.0, "calc_p": False},
            {"key": "room_a", "p": 0.0, "t": 20.0, "calc_p": True, "v": 40.0},
            {"key": "room_b", "p": 0.0, "t": 20.0, "calc_p": True, "v": 40.0},
        ],
        "ventilation_branches": openings,
        "thermal_branches": [],
    }


@pytest.mark.physics
@requires_solver
def test_stiff_pressure_network_converges_finite(solver_workdir):
    out, art = run_from_raw(
        raw_config=_raw_stiff_pressure_network(),
        run_id="stiff_pressure",
        tmp_base_dir=solver_workdir,
        add_surface=False,
        add_capacity=False,
        log_verbosity=1,
    )

    _pk, press_rows = read_series_matrix(art, out, "vent_pressure")
    flow_keys, flow_rows = read_series_matrix(art, out, "vent_flow_rate")
    assert press_rows and flow_rows
    for t, row in enumerate(press_rows):
        assert_all_finite(row, label=f"vent_pressure[{t}]")
    for t, row in enumerate(flow_rows):
        assert_all_finite(row, label=f"vent_flow_rate[{t}]")
        assert any(abs(v) > 1e-9 for v in row), f"all flows ~0 at t={t}"

    for t, row in enumerate(flow_rows):
        d = dict(zip(flow_keys, row))
        q_in = sum(v for k, v in d.items() if k.startswith("out_h->room_a"))
        q_mid = sum(v for k, v in d.items() if k.startswith("room_a->room_b"))
        q_out = sum(v for k, v in d.items() if k.startswith("room_b->out_l"))
        assert abs(q_in - q_mid) <= tol.VENT_MASS_BALANCE_ABS, f"imbalance a at t={t}: {q_in=} {q_mid=}"
        assert abs(q_mid - q_out) <= tol.VENT_MASS_BALANCE_ABS, f"imbalance b at t={t}: {q_mid=} {q_out=}"

    assert_artifact_no_nan_inf(art, out)
    assert_convergence_from_log(
        read_solver_log(art, out),
        expect_thermal=False,
        expect_pressure=True,
    )

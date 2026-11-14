"""A weak gap must satisfy flow balance even when J^T r is already tiny."""
import pytest

from .conftest import requires_solver
from .helpers import run_from_raw, read_series_matrix, read_solver_log


@pytest.mark.physics
@requires_solver
@pytest.mark.parametrize("a", [1e-6, 1e-7, 1e-8])
@pytest.mark.parametrize("scaled_flow", [False, True])
def test_weak_gap_gradient_does_not_accept_stagnant_pressure(solver_workdir, a, scaled_flow):
    # Scale the imposed flow with conductance: the solution remains at 100/121 Pa
    # across cases, while the gradient shrinks quadratically with conductance.
    tolerance = a * 0.01 if scaled_flow else 1e-7
    flow = a * 10 if scaled_flow else 1e-5
    raw = {
        "simulation": {
            "index": {"start": "2026-01-01T00:00:00", "end": "2026-01-01T01:00:00", "timestep": 3600, "length": 2},
            "calc_flag": {"p": True, "t": False, "x": False, "c": False},
            "tolerance": {"ventilation": tolerance},
        },
        "nodes": [{"key": "out", "p": 0.0, "t": 20.0, "calc_p": False},
                  {"key": "room", "p": -1.0, "t": 20.0, "calc_p": True, "v": 40.0}],
        "ventilation_branches": [{"key": "out->room", "type": "gap", "a": a, "n": 2.0},
                                 {"key": "room->out", "vol": [flow, flow * 1.1]}],
        "thermal_branches": [],
    }
    output, art = run_from_raw(raw_config=raw, run_id="weak_gradient", tmp_base_dir=solver_workdir,
                               add_surface=False, add_capacity=False, log_verbosity=1)
    keys, rows = read_series_matrix(art, output, "vent_flow_rate")
    for row in rows:
        q = dict(zip(keys, row))
        assert abs(q["out->room"] - q["room->out"]) <= tolerance
    log = read_solver_log(art, output)
    assert "勾配停止を精密化" in log
    assert "フォールバック開始" not in log

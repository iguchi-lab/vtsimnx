"""Weak paths must retain physical acceptance with the large-network backend."""
import numpy as np
import pytest

from .conftest import requires_solver
from .helpers import read_series_matrix, run_from_raw
from tools.benchmark_pressure import pressure_case


@pytest.mark.physics
@requires_solver
def test_sparse_chain_weak_conductance(solver_workdir):
    raw, expected = pressure_case(64, 3, conductance=1e-8)
    raw["simulation"]["tolerance"]["ventilation"] = 1e-12
    output, art = run_from_raw(raw_config=raw, run_id="sparse_weak_chain",
                               tmp_base_dir=solver_workdir, add_capacity=False, add_surface=False)
    keys, rows = read_series_matrix(art, output, "vent_pressure")
    actual = np.asarray(rows)[:, [keys.index(f"r{i}") for i in range(64)]]
    assert np.max(abs(actual-expected)) < .002
    assert output["metrics"]["pressure_fallback_count"] == 0

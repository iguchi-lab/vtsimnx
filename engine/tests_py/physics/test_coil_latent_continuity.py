"""Issue #24: dry/wet coil transitions must not cause an outer-loop cycle."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.solver_runner as sr
from .conftest import requires_solver
from .helpers import (
    assert_artifact_no_nan_inf,
    assert_convergence_from_log,
    read_series,
    read_solver_log,
    run_from_raw,
)


@pytest.fixture
def coil_workdir(solver_workdir, monkeypatch):
    binary = Path(os.environ.get("VTSIMNX_TEST_SOLVER", str(sr.SOLVER_EXE))).resolve()
    if not binary.is_file():
        pytest.skip("solver binary not found")
    monkeypatch.setattr(sr, "SOLVER_EXE", binary)
    return solver_workdir


def _raw_coil_case(outside_x, *, moist=False, duct=False):
    length = len(outside_x) if isinstance(outside_x, list) else 1
    spec = {"latent_method": "coil_aoaf"}
    if duct:
        # Same equipment data as app/test_json/config_01.json; all volumes are m3/s.
        spec.update({
            "Q": {"cooling": {"min": 2.682626, "mid": 3.832323, "rtd": 7.664646},
                  "heating": {"min": 2.706897, "mid": 3.866996, "rtd": 7.733992}},
            "P": {"cooling": {"rtd": 2.41787}, "heating": {"rtd": 2.056912}},
            "P_fan": {"cooling": {"mid": 0.188513, "rtd": 0.240326},
                      "heating": {"mid": 0.188981, "rtd": 0.241263}},
            "V_inner": {"cooling": {"mid": 0.349611, "dsgn": 0.361468, "rtd": 0.457554},
                        "heating": {"mid": 0.350587, "dsgn": 0.363011, "rtd": 0.459508}},
        })
    return {
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": f"2000-01-01T{length - 1:02d}:00:00",
                      "timestep": 3600, "length": length},
            "tolerance": {"thermal": 1e-3, "convergence": 1e-6, "coupling_humidity": 1e-8},
            "calc_flag": {"p": False, "t": True, "x": True, "c": False},
            "coupling": {"moisture_enabled": True, "moist_enthalpy_enabled": moist},
        },
        "nodes": [
            {"key": "outside", "t": 32.0, "x": outside_x},
            {"key": "room", "t": 27.5, "calc_t": True, "x": 0.0082, "calc_x": True, "v": 40.0},
        ],
        "thermal_branches": [{"key": "outside->room", "conductance": 1000.0}],
        "ventilation_branches": [{"key": "outside->room", "vol": 0.05},
                                 {"key": "room->outside", "vol": 0.05}],
        "aircon": [{"key": "AC", "set": "room", "outside": "outside", "pre_temp": 27.5,
                    "mode": "COOLING", "model": "DUCT_CENTRAL" if duct else "IDEAL",
                    "vol": 0.3, "ac_spec": spec}],
    }


def _verify_case(raw, workdir):
    out, art = run_from_raw(raw_config=raw, run_id="coil-dew-boundary", tmp_base_dir=workdir,
                           add_surface=False, add_aircon=True, add_capacity=False,
                           add_moisture_capacity=False, log_verbosity=2)
    room_t = read_series(art, out, "thermal_temperature", "room")
    room_x = read_series(art, out, "humidity_x", "room")
    supply_x = read_series(art, out, "humidity_x", "AC")
    latent_w = read_series(art, out, "aircon_latent_heat", "AC")
    assert len(room_t) == raw["simulation"]["index"]["length"]
    assert all(abs(t - 27.5) <= 1e-3 for t in room_t), room_t
    assert all(0.0 <= xs <= xr + 1e-8 for xs, xr in zip(supply_x, room_x))
    assert all(q >= 0.0 for q in latent_w) and max(latent_w) > 0.0, latent_w
    assert_artifact_no_nan_inf(art, out)
    log = read_solver_log(art, out)
    assert "regime=coil_wet" in log
    assert_convergence_from_log(log, expect_thermal=True)


@pytest.mark.physics
@requires_solver
@pytest.mark.parametrize("outside_x", [0.009, 0.010, 0.012])
@pytest.mark.parametrize("moist", [False, True])
def test_coil_humidity_feedback_converges(coil_workdir, outside_x, moist):
    # Before the fix, the latent load alternated between zero and ~1.3 kW,
    # exhausting 100 outer iterations despite converged inner solves.
    _verify_case(_raw_coil_case(outside_x, moist=moist), coil_workdir)


@pytest.mark.physics
@requires_solver
@pytest.mark.parametrize("moist", [False, True])
def test_duct_coil_tracks_changing_humidity(coil_workdir, moist):
    _verify_case(_raw_coil_case([0.009, 0.012, 0.016, 0.010, 0.009, 0.012],
                               moist=moist, duct=True), coil_workdir)

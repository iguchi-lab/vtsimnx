"""A heating load increase must update the fan before accepting a thermal candidate."""
from pathlib import Path
import os

import pytest

from app.builder import build_config
import app.solver_runner as sr
from tests_py.physics.helpers import read_series


@pytest.fixture
def heating_solver(tmp_path, monkeypatch):
    binary = Path(os.environ.get("VTSIMNX_TEST_SOLVER", str(sr.SOLVER_EXE))).resolve()
    if not binary.is_file():
        pytest.skip("solver binary not found")
    monkeypatch.setattr(sr, "SOLVER_EXE", binary)
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)


def heating_fan_case(restart=False):
    # First reach the minimum fan speed at a 350 W load, then demand 7 kW.
    # The old ordering tried to heat at that stale low speed and exceeded 120 C.
    outside = [19.0, 21.0, 0.0] if restart else [19.0, 0.0]
    modes = ["HEATING", "OFF", "HEATING"] if restart else ["HEATING"] * 2
    return {
        "simulation": {
            "index": {"start": "2026-01-01T00:00:00",
                      "end": f"2026-01-01T{len(outside)-1:02d}:00:00",
                      "length": len(outside), "timestep": 3600},
            "calc_flag": {"p": True, "t": True, "x": False, "c": False},
            "tolerance": {"ventilation": 1e-5, "thermal": 1e-3, "convergence": 1e-3},
            "log": {"verbosity": 1},
        },
        "nodes": [{"key": "outside", "t": outside},
                  {"key": "room", "t": 20.0, "calc_t": True}],
        "thermal_branches": [{"key": "outside->room", "conductance": 350.0}],
        "ventilation_branches": [],
        "aircon": [{
            "key": "AC", "set": "room", "outside": "outside",
            "pre_temp": 20.0, "mode": modes, "model": "DUCT_CENTRAL",
            "p_max": 440.0, "p1": 200.0, "q1": 0.3777777778, "q_max": 0.5222222222,
            "ac_spec": {
                "Q": {"heating": {"min": 2.706897, "mid": 3.866996, "rtd": 7.733992}},
                "P": {"heating": {"rtd": 2.056912}},
                "P_fan": {"heating": {"mid": 0.188981, "rtd": 0.241263}},
                "V_inner": {"heating": {"mid": 0.350587, "dsgn": 0.363011, "rtd": 0.459508}},
            },
        }],
    }


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("backend", ["klu", "eigen"])
def test_heating_load_increase_from_minimum_fan(heating_solver, monkeypatch, restart, backend):
    monkeypatch.setenv("VTSIMNX_THERMAL_DIRECT_LU", backend)
    raw = heating_fan_case(restart)
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=True,
                         add_capacity=False, add_moisture_capacity=False)
    out = sr.run_solver(built)
    assert out["status"] == "ok", out
    art = sr.resolve_artifact_path(out["artifact_dir"])
    room = read_series(art, out, "thermal_temperature", "room")
    supply = read_series(art, out, "thermal_temperature", "AC")
    heat = read_series(art, out, "aircon_sensible_heat", "AC")
    assert room[-1] == pytest.approx(20.0, abs=1e-3), room
    assert 20.0 < supply[-1] < 60.0, supply
    assert heat[-1] == pytest.approx(7000.0, abs=2.0), heat
    if restart:
        assert heat[1] == pytest.approx(0.0, abs=1e-3), heat
    assert "暖房ファン先行補正" in (art / "solver.log").read_text()


def test_fan_retry_respects_coupling_limit(heating_solver):
    built = build_config(heating_fan_case(), output_path=None, add_surface=False,
                         add_aircon=True, add_capacity=False, add_moisture_capacity=False)
    built["simulation"]["iteration"] = {"max_coupling": 2}
    out = sr.run_solver(built)
    assert out["status"] == "error", out
    assert out["error_code"] == "coupling_max_iterations", out
    art = sr.resolve_artifact_path(out["artifact_dir"])
    assert "暖房ファン先行補正" in (art / "solver.log").read_text()


@pytest.mark.parametrize("backend", ["klu", "eigen"])
def test_minimum_fan_does_not_freeze_a_growing_distribution_load(heating_solver, monkeypatch, backend):
    # An intentionally uneven distribution exposes the minimum-flow latch:
    # lowering the fan increases supply temperature and the return-zone load.
    raw = heating_fan_case()
    raw["simulation"]["index"].update(end="2026-01-01T00:00:00", length=1)
    raw["nodes"] = [
        {"key": "outside", "t": 18.0},
        {"key": "room", "t": 20.0, "calc_t": True, "calc_p": True},
        {"key": "return", "t": 20.0, "calc_t": True, "p": 0.0},
        {"key": "supply", "t": 20.0, "calc_t": True, "calc_p": True},
    ]
    raw["thermal_branches"] = [
        {"key": "outside->room", "conductance": 150.0},
        {"key": "outside->return", "conductance": 40.0},
    ]
    raw["ventilation_branches"] = [
        {"key": "supply->room", "alpha": 0.65, "area": 0.01},
        {"key": "room->return", "alpha": 0.65, "area": 2.0},
        {"key": "supply->return", "alpha": 0.65, "area": 0.25},
    ]
    raw["aircon"][0].update({"in": "return", "out": "supply", "mode": "HEATING"})
    monkeypatch.setenv("VTSIMNX_THERMAL_DIRECT_LU", backend)
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=True,
                         add_capacity=False, add_moisture_capacity=False)
    out = sr.run_solver(built)
    assert out["status"] == "ok", out
    art = sr.resolve_artifact_path(out["artifact_dir"])
    room = read_series(art, out, "thermal_temperature", "room")[0]
    return_temp = read_series(art, out, "thermal_temperature", "return")[0]
    supply = read_series(art, out, "thermal_temperature", "supply")[0]
    heat = read_series(art, out, "aircon_sensible_heat", "AC")[0]
    assert room == pytest.approx(20.0, abs=1e-3)
    assert 20.0 < supply < 100.0
    # Accepted coil heat must balance the two external conduction losses.
    assert heat == pytest.approx(150.0 * (room - 18.0) + 40.0 * (return_temp - 18.0), abs=0.1)
    assert heat > 2.0 * 773.3992  # It must not be replaced by the stale minimum.

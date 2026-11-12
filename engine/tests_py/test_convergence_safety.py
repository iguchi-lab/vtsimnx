"""End-to-end regressions: small steps and rejected states are not solutions."""
import os
from pathlib import Path

import pytest

from app.builder import build_config
import app.solver_runner as sr


@pytest.fixture
def isolated_solver(tmp_path, monkeypatch):
    exe = Path(os.environ.get("VTSIMNX_TEST_SOLVER", str(sr.SOLVER_EXE))).resolve()
    if not exe.is_file():
        pytest.skip("solver binary not found")
    monkeypatch.setattr(sr, "SOLVER_EXE", exe)
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    return tmp_path


def run_case(*, coupled=True, source_temp=1000.0, temperature_alpha=1.0,
             humidity_alpha=1.0, source_x=0.01, humidity_enabled=True):
    raw = {
        "simulation": {
            "index": {"start": "2026-01-01T00:00:00", "end": "2026-01-01T00:00:00",
                      "timestep": 3600, "length": 1},
            "tolerance": {"ventilation": 1e-5, "thermal": 1e-3, "convergence": 1e-3},
            "calc_flag": {"p": False, "t": True, "x": humidity_enabled, "c": False},
            "coupling": {"moisture_enabled": coupled, "temperature_relaxation": temperature_alpha,
                         "humidity_relaxation": humidity_alpha},
            "log": {"verbosity": 1},
        },
        "nodes": [
            {"key": "outside", "t": source_temp, "x": source_x},
            {"key": "room", "t": 20.0, "calc_t": True, "x": 0.01,
             "calc_x": humidity_enabled},
        ],
        "thermal_branches": [{"key": "outside->room", "conductance": 10.0}],
        "ventilation_branches": [
            {"key": "outside->room", "vol": 0.1},
            {"key": "room->outside", "vol": 0.1},
        ],
    }
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=False,
                         add_capacity=False, add_moisture_capacity=False)
    # Bound runtime and distinguish false convergence from mere slow convergence.
    built["simulation"]["iteration"] = {"max_coupling": 3}
    return sr.run_solver(built)


@pytest.mark.parametrize("coupled,humidity_enabled", [(True, True), (False, True), (True, False)])
def test_unphysical_temperature_never_returns_ok(isolated_solver, coupled, humidity_enabled):
    out = run_case(coupled=coupled, humidity_enabled=humidity_enabled)
    assert out["status"] == "error", out
    assert out["error_code"] == "thermal_not_converged", out
    assert "node=room" in out["error"]


@pytest.mark.parametrize("coupled", [True, False])
def test_unphysical_humidity_never_returns_ok(isolated_solver, coupled):
    out = run_case(coupled=coupled, source_temp=20.0, source_x=0.8)
    assert out["status"] == "error", out
    assert out["error_code"] == "humidity_not_converged", out
    assert "node=room" in out["error"]


@pytest.mark.parametrize("field", ["temperature", "humidity"])
def test_tiny_relaxation_does_not_hide_large_defect(isolated_solver, field):
    args = {"source_temp": 20.0}
    if field == "temperature":
        args.update(source_temp=30.0, temperature_alpha=1e-6)
    else:
        args.update(source_x=0.02, humidity_alpha=1e-6)
    out = run_case(**args)
    assert out["status"] == "error", out
    assert out["error_code"] == "coupling_max_iterations", out


def test_physical_equilibrium_still_succeeds(isolated_solver):
    out = run_case(source_temp=20.0)
    assert out["status"] == "ok", out


def test_latent_trace_identifies_branch_and_writer(isolated_solver):
    from tests_py.physics.test_hvac_onoff import _raw_hvac_onoff_case
    raw = _raw_hvac_onoff_case()
    raw["simulation"]["log"] = {"verbosity": 2}
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=True, add_capacity=True)
    out = sr.run_solver(built)
    assert out["status"] == "ok", out
    log = (sr.resolve_artifact_path(out["artifact_dir"]) / "solver.log").read_text()
    assert "latent_boundary stage=capacity key=AC1" in log
    assert "regime=non_cooling_passthrough" in log
    assert all(key in log for key in ("x_in=", "x_old=", "x_new=", "removal_kg_s=", "recompute="))


def test_saved_failure_replays_with_same_error(isolated_solver):
    import json
    import subprocess
    out = run_case(source_temp=1000.0)
    assert out["status"] == "error"
    bundle = next((isolated_solver / "work/runs").glob("*/replay"))
    script = Path(__file__).resolve().parents[1] / "tools/replay_failure.py"
    import sys
    result = subprocess.run([sys.executable, str(script), str(bundle), "--solver", str(sr.SOLVER_EXE),
                             "--out", str(isolated_solver / "reproduced")], capture_output=True, text=True)
    assert result.returncode != 0
    replayed = json.loads((isolated_solver / "reproduced/output.json").read_text())
    assert replayed["error_code"] == out["error_code"]
    assert replayed["error"] == out["error"]


@pytest.mark.parametrize("backend", ["klu", "eigen"])
def test_cooling_setpoint_does_not_cycle(isolated_solver, monkeypatch, backend):
    monkeypatch.setenv("VTSIMNX_THERMAL_DIRECT_LU", backend)
    from tests_py.physics.test_hvac_onoff import _raw_hvac_onoff_case
    raw = _raw_hvac_onoff_case()
    raw["nodes"][0]["t"] = 32.0
    raw["nodes"][1]["t"] = 30.0
    ac = raw["aircon"][0]
    ac["mode"] = ["COOLING"] * 6
    ac["pre_temp"] = 26.0
    ac["ac_spec"]["latent_method"] = "none"
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=True, add_capacity=True)
    out = sr.run_solver(built)
    assert out["status"] == "ok", out

    from tests_py.physics.helpers import read_series
    art = sr.resolve_artifact_path(out["artifact_dir"])
    room_t = read_series(art, out, "thermal_temperature", "room")
    # The first step has enough cooling load to exceed minimum capacity.
    # A pattern-only symmetry check used to accept 12.47 C instead of 26 C.
    assert room_t[0] == pytest.approx(26.0, abs=1e-3), room_t
    assert all(0.0 < temp < 32.0 for temp in room_t), room_t

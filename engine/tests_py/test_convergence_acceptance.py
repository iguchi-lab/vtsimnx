"""Physical acceptance regressions: conservation, capacity, retries and raw defects."""
import os
from pathlib import Path

import pytest

from app.builder import build_config
import app.solver_runner as sr
from tests_py.physics.helpers import read_series
from tests_py.test_heating_fan_coupling import heating_fan_case


@pytest.fixture(autouse=True)
def isolated_solver(tmp_path, monkeypatch):
    binary = Path(os.environ.get("VTSIMNX_TEST_SOLVER", str(sr.SOLVER_EXE))).resolve()
    if not binary.is_file():
        pytest.skip("solver binary not found")
    monkeypatch.setattr(sr, "SOLVER_EXE", binary)
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)


def build(raw, aircon=True):
    return build_config(raw, output_path=None, add_surface=False, add_aircon=aircon,
                        add_capacity=False, add_moisture_capacity=False)


def result(built):
    out = sr.run_solver(built)
    assert out["status"] == "ok", out
    art = sr.resolve_artifact_path(out["artifact_dir"])
    log = (art / "solver.log").read_text()
    assert "最終状態整合" in log
    return out, art


def fixed_flow(ac, flow=0.363011):
    for key in ("p_max", "p1", "q1", "q_max"):
        ac.pop(key, None)
    ac["vol"] = flow


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("mode", ["HEATING", "AUTO", "OFF"])
@pytest.mark.parametrize("generation", [0.0, 1e-6, 1e-5])
def test_closed_room_preserves_generated_water(coupled, mode, generation):
    raw = heating_fan_case()
    raw["simulation"]["index"].update(end="2026-01-01T00:00:00", length=1)
    raw["simulation"]["calc_flag"]["x"] = True
    raw["simulation"]["coupling"] = {"moisture_enabled": coupled}
    raw["simulation"]["tolerance"]["coupling_humidity"] = 1e-8
    raw["nodes"][0].update(t=19.0, x=0.01)
    raw["nodes"][1].update(x=0.01, calc_x=True, v=30.0)
    raw["aircon"][0]["mode"] = mode
    fixed_flow(raw["aircon"][0], 0.0363011)
    raw["humidity_source"] = [{"key": "generation", "room": "room",
                               "generation_rate": generation}]
    out, art = result(build(raw))
    x = read_series(art, out, "humidity_x", "room")[0]
    # Closed 36 kg dry-air inventory: recirculation cannot remove any water.
    assert x == pytest.approx(0.01 + generation * 3600 / (1.2 * 30), abs=1e-8)
    if mode in ("HEATING", "AUTO"):
        supply_x = read_series(art, out, "humidity_x", "AC")[0]
        assert supply_x == pytest.approx(x, abs=1e-8)


@pytest.mark.parametrize("mode", ["HEATING", "COOLING"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("backend", ["klu", "eigen"])
def test_load_jump_updates_all_airflow_paths(monkeypatch, mode, fixed, restart, backend):
    monkeypatch.setenv("VTSIMNX_THERMAL_DIRECT_LU", backend)
    raw = heating_fan_case(restart)
    ac = raw["aircon"][0]
    target = 20.0
    if mode == "COOLING":
        target = 26.0
        raw["nodes"][0]["t"] = [27.0, 25.0, 46.0] if restart else [27.0, 46.0]
        raw["nodes"][1]["t"] = target
        ac["pre_temp"] = target
        ac["mode"] = ["COOLING", "OFF", "COOLING"] if restart else ["COOLING"] * 2
        for group in ac["ac_spec"].values():
            if isinstance(group, dict) and "heating" in group:
                group["cooling"] = group.pop("heating")
        ac["ac_spec"]["latent_method"] = "none"
    if fixed:
        fixed_flow(ac)
        raw["simulation"]["calc_flag"]["p"] = False
    out, art = result(build(raw))
    room = read_series(art, out, "thermal_temperature", "room")
    supply = read_series(art, out, "thermal_temperature", "AC")
    heat = read_series(art, out, "aircon_sensible_heat", "AC")
    assert room[-1] == pytest.approx(target, abs=1e-3)
    assert -10 < supply[-1] < 60
    assert abs(heat[-1]) == pytest.approx(7000.0, abs=2.0)
    if restart:
        assert heat[1] == pytest.approx(0.0, abs=1e-3)
    assert "空調風量先行補正" in (art / "solver.log").read_text()


def test_remote_room_setpoint_cannot_bypass_capacity_limit():
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
    ac = raw["aircon"][0]
    ac.update({"in": "return", "out": "supply", "mode": "HEATING"})
    ac["ac_spec"]["Q"]["heating"] = {"min": 0.01, "mid": 0.05, "rtd": 0.1, "max": 0.1}
    out, art = result(build(raw))
    heat = read_series(art, out, "aircon_sensible_heat", "AC")[0]
    room = read_series(art, out, "thermal_temperature", "room")[0]
    return_t = read_series(art, out, "thermal_temperature", "return")[0]
    assert 0.0 < heat <= 100.1
    assert room < 20.0  # The unmet load must not be hidden by exceeding Q.max.
    assert heat == pytest.approx(150 * (room - 18) + 40 * (return_t - 18), abs=0.02)


def phase_change_case(alpha):
    raw = {
        "simulation": {
            "index": {"start": "2026-01-01T00:00:00", "end": "2026-01-01T00:00:00",
                      "length": 1, "timestep": 3600},
            "calc_flag": {"p": False, "t": True, "x": True, "c": False},
            "tolerance": {"thermal": 1e-3, "convergence": 1e-3, "coupling_humidity": 1e-8},
            "coupling": {"moisture_enabled": True, "latent_coupling_mode": "from_phase_change",
                         "latent_relaxation": alpha, "latent_absolute_tolerance_w": 1e-3,
                         "latent_relative_tolerance": 1e-3},
            "log": {"verbosity": 1},
        },
        "nodes": [{"key": "outside", "t": 20.0, "x": 0.01},
                  {"key": "room", "t": 20.0, "calc_t": True, "x": 0.01, "calc_x": True, "v": 100.0},
                  {"key": "mat", "t": 20.0, "calc_t": True, "x": 0.002, "calc_x": True}],
        "thermal_branches": [{"key": "outside->room", "conductance": 10.0},
                             {"key": "outside->mat", "conductance": 10.0},
                             {"key": "mat->room", "conductance": 0.0, "moisture_conductance": 0.002}],
        "ventilation_branches": [],
    }
    built = build(raw, aircon=False)
    next(n for n in built["nodes"] if n["key"] == "mat")["moisture_capacity"] = 50.0
    return built


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_phase_change_heat_is_balanced_before_acceptance(alpha):
    out, art = result(phase_change_case(alpha))
    mat_t = read_series(art, out, "thermal_temperature", "mat")[0]
    # Independent implicit two-inventory exchange, L=2.5 MJ/kg.
    final_x_difference = 0.008 / (1 + 3600 * 0.002 * (1 / 50 + 1 / 120))
    latent_w = 2_500_000 * 0.002 * final_x_difference
    assert 10 * (mat_t - 20) == pytest.approx(latent_w, abs=0.04)


def test_tiny_relaxation_is_not_false_convergence():
    built = phase_change_case(1e-6)
    built["simulation"]["iteration"] = {"max_coupling": 30}
    out = sr.run_solver(built)
    assert out["status"] == "error", out
    assert out["error_code"] == "coupling_max_iterations", out

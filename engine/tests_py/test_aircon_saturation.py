"""Outlet saturation and temperature-driven dehumidification in serialized results."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from app.builder import build_config
import app.solver_runner as sr
from tests_py.physics.helpers import read_series
from tests_py.physics.test_coil_latent_continuity import _raw_coil_case
from vtsimnx.archenv.archenv import (
    saturation_vapor_pressure_pa,
    vapor_pressure_from_humidity_ratio_pa,
)


@pytest.fixture(autouse=True)
def isolated_solver(tmp_path, monkeypatch):
    binary = Path(os.environ.get("VTSIMNX_TEST_SOLVER", str(sr.SOLVER_EXE))).resolve()
    if not binary.is_file():
        pytest.skip("solver binary not found")
    monkeypatch.setattr(sr, "SOLVER_EXE", binary)
    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)


def run(raw):
    built = build_config(raw, output_path=None, add_surface=False, add_aircon=True,
                         add_capacity=False, add_moisture_capacity=False)
    built["simulation"].setdefault("log", {})["verbosity"] = 1
    out = sr.run_solver(built)
    assert out["status"] == "ok", out
    art = sr.resolve_artifact_path(out["artifact_dir"])
    log = (art / "solver.log").read_text()
    assert "最終状態整合" in log
    errors = [line for line in log.splitlines() if "[ERROR]" in line]
    assert all("DirectT: LLT 因子分解に失敗（非SPD/悪条件）" in line for line in errors), errors

    def series(name, key):
        return np.asarray(read_series(art, out, name, key))

    return series


@pytest.mark.parametrize("method", ["rh95", "bf", "coil_aoaf", "none", "ideal_rh60"])
@pytest.mark.parametrize("coupled,moist", [(False, False), (True, False), (True, True)])
def test_actual_supply_stays_unsaturated_with_loose_tolerance(method, coupled, moist):
    raw = _raw_coil_case([0.024, 0.024, 0.024, 0.018, 0.026], duct=True, moist=moist)
    raw["simulation"]["tolerance"]["coupling_humidity"] = 1e-3
    raw["simulation"]["coupling"]["moisture_enabled"] = coupled
    ac = raw["aircon"][0]
    ac["mode"] = ["COOLING", "OFF", "COOLING", "COOLING", "COOLING"]
    ac["ac_spec"]["latent_method"] = "none" if method == "ideal_rh60" else method
    if method == "ideal_rh60":
        ac["pre_rh"] = 60.0
    # Separate return room and zero-volume discharge plenum, as in the annual
    # whole-house pressure network. A fixed outgoing plenum flow would remain
    # nonzero while the fan is OFF, so use pressure-driven ducts throughout.
    raw["simulation"]["calc_flag"]["p"] = True
    raw["nodes"][1].update(calc_p=True, p=0.0)
    raw["nodes"].extend([
        {"key": "return", "t": 27.5, "x": 0.0082, "calc_t": True,
         "calc_x": True, "calc_p": True, "v": 20.0},
        {"key": "supply", "t": 27.5, "x": 0.0082, "calc_t": True,
         "calc_x": True, "calc_p": True, "v": 0.0},
    ])
    ac.update({"in": "return", "out": "supply", "p_max": 440.0, "p1": 200.0,
               "q1": 0.3777777778, "q_max": 0.5222222222})
    ac.pop("vol")
    raw["ventilation_branches"].extend([
        {"key": "supply->room", "alpha": 0.65, "area": 1.0},
        {"key": "room->return", "alpha": 0.65, "area": 1.0},
    ])
    s = run(raw)
    for key in ("AC", "supply"):
        t, x = s("thermal_temperature", key), s("humidity_x", key)
        rh = 100 * vapor_pressure_from_humidity_ratio_pa(x) / saturation_vapor_pressure_pa(t)
        assert np.max(rh) <= 100.0, (method, key, rh)
    assert np.max(s("aircon_latent_heat", "AC")) > 0
    assert s("aircon_latent_heat", "AC")[1] == 0


@pytest.mark.parametrize("method", ["none", "rh95", "bf", "coil_aoaf"])
@pytest.mark.parametrize("moist", [False, True])
def test_temperature_priority_rh_is_only_an_evaluation_threshold(method, moist):
    raw = _raw_coil_case([0.018] * 4, duct=True, moist=moist)
    raw["nodes"][0]["t"] = [26.1, 26.0, 32.0, 26.1]
    raw["nodes"][1].update(t=26.0, x=0.018)
    ac = raw["aircon"][0]
    ac.update(model="RAC", pre_temp=26.0, mode=["COOLING", "OFF", "COOLING", "COOLING"])
    ac["ac_spec"]["Q"]["cooling"]["max"] = 20.0
    ac["ac_spec"]["latent_method"] = method
    baseline = run(raw)
    for rh in (40.0, 60.0, 80.0):
        requested = deepcopy(raw)
        requested["aircon"][0].update(pre_rh=rh, control_mode="temperature_priority")
        s = run(requested)
        for artifact in ("thermal_temperature", "humidity_x", "aircon_sensible_heat",
                         "aircon_latent_heat", "aircon_power", "aircon_reheat"):
            np.testing.assert_allclose(s(artifact, "AC"), baseline(artifact, "AC"), rtol=0, atol=1e-7)
        assert s("aircon_sensible_heat", "AC")[0] > 0
        # Small nighttime cooling stays above the dew point: no independent drying.
        assert s("aircon_latent_heat", "AC")[0] == pytest.approx(0.0, abs=1e-9)
        assert s("humidity_x", "AC")[0] == pytest.approx(s("humidity_x", "room")[0], abs=1e-7)
        assert s("aircon_latent_heat", "AC")[2] > 0
        assert s("aircon_power", "AC")[2] > 0
        assert max(s("aircon_reheat", "AC")) == 0
        assert s("aircon_power", "AC")[1] == 0
        assert s("aircon_latent_heat", "AC")[1] == 0
        assert s("aircon_humidity_unmet", "AC")[0] == pytest.approx(
            max(0.0, s("aircon_room_rh", "AC")[0] - rh), abs=1e-5)


@pytest.mark.parametrize("method", ["coil_aoaf", "ideal_rh60"])
@pytest.mark.parametrize("coupled", [False, True])
def test_discharge_plenum_stays_unsaturated_with_small_flow_residual(method, coupled):
    raw = _raw_coil_case(0.026, duct=True)
    raw["simulation"]["coupling"]["moisture_enabled"] = coupled
    raw["nodes"].extend([
        {"key": "return", "t": 30.0, "x": 0.026},
        {"key": "supply", "t": 27.5, "x": 0.0082,
         "calc_t": True, "calc_x": True, "v": 0.0},
    ])
    ac = raw["aircon"][0]
    # Use fixed-flow RAC to isolate water balance from automatic fan control.
    # The pressure-driven cases above separately cover DUCT_CENTRAL.
    ac.update({"in": "return", "out": "supply", "model": "RAC"})
    ac["ac_spec"]["Q"]["cooling"]["max"] = 20.0
    ac["ac_spec"]["latent_method"] = "none" if method == "ideal_rh60" else method
    if method == "ideal_rh60":
        ac["pre_rh"] = 60.0
    # A tiny fixed imbalance reproduces the pressure solver's accepted mass
    # residual. Water conservation gives x_supply = x_AC * q_in / q_out.
    raw["ventilation_branches"].extend([
        {"key": "supply->room", "vol": 0.3 * (1 - 1e-5)},
        {"key": "room->return", "vol": 0.3 * (1 - 1e-5)},
    ])
    s = run(raw)
    for key in ("AC", "supply"):
        t, x = s("thermal_temperature", key), s("humidity_x", key)
        rh = 100 * vapor_pressure_from_humidity_ratio_pa(x) / saturation_vapor_pressure_pa(t)
        assert np.max(rh) <= 100.0, (method, key, rh)
    assert s("humidity_x", "supply")[0] == pytest.approx(
        s("humidity_x", "AC")[0] / (1 - 1e-5), abs=2e-9)
    assert s("aircon_latent_heat", "AC")[0] > 0

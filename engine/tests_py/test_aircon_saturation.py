"""Outlet saturation and controlled-room ideal RH, including serialized results."""
import os
from pathlib import Path

import numpy as np
import pytest

from app.builder import build_config
import app.solver_runner as sr
from tests_py.physics.helpers import read_series
from tests_py.physics.test_coil_latent_continuity import _raw_coil_case
from vtsimnx.archenv.archenv import (
    humidity_ratio_from_rh,
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
    assert "最終状態整合" in (art / "solver.log").read_text()

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


def test_ideal_humidity_uses_requested_room_temperature_not_return_air():
    raw = _raw_coil_case(0.022, duct=True)
    raw["thermal_branches"][0]["conductance"] = 50.0
    raw["nodes"].append({"key": "return", "t": 31.0, "x": 0.022})
    raw["ventilation_branches"].append({"key": "room->return", "vol": 0.3})
    ac = raw["aircon"][0]
    ac.update({"in": "return", "pre_temp": 27.5, "pre_rh": 60.0})
    ac["ac_spec"].pop("V_inner")
    ac["ac_spec"]["latent_method"] = "rh95"
    s = run(raw)
    # At this mild sensible load the outlet remains above the target dew point.
    expected = humidity_ratio_from_rh(27.5, 60.0)
    assert s("humidity_x", "AC")[0] == pytest.approx(expected, abs=1e-7)
    # Capacity-limited actual room temperature must not redefine the ideal goal.
    assert s("thermal_temperature", "room")[0] > 27.5

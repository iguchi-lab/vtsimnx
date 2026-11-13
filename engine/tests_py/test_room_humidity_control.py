"""Room humidistat modes: independent room balances and infeasible targets."""
from copy import deepcopy

import numpy as np
import pytest

from app.builder import build_config
from app.builder.validate import ValidationError
from app.schemas.config import RawSimConfig
from tests_py.test_aircon_saturation import isolated_solver, run  # noqa: F401
from tests_py.physics.test_coil_latent_continuity import _raw_coil_case
from vtsimnx.archenv.archenv import humidity_ratio_from_rh


def case(mode="temperature_humidity", *, moist=False, outside_t=26.0, x=0.018, count=6):
    raw = _raw_coil_case([x] * count, duct=True, moist=moist)
    raw["nodes"][0]["t"] = outside_t
    raw["nodes"][1].update(t=26.0, x=x)
    ac = raw["aircon"][0]
    ac.update(model="RAC", pre_temp=26.0, pre_rh=60.0, vol=0.1,
              control_mode=mode, max_overcooling=2.0)
    ac["ac_spec"]["Q"]["cooling"]["max"] = 20.0
    ac["ac_spec"]["Q"]["cooling"]["min"] = 0.01
    return raw


@pytest.mark.parametrize("mode", ["temperature_priority", "humidity_priority", "temperature_humidity"])
@pytest.mark.parametrize("moist", [False, True])
def test_humidity_demand_without_sensible_demand(mode, moist):
    s = run(case(mode, moist=moist))
    t, rh = s("thermal_temperature", "room"), s("aircon_room_rh", "AC")
    if mode == "temperature_priority":
        assert np.allclose(t, 26.0, atol=0.01)
        assert min(rh) > 70.0
        assert max(s("aircon_latent_heat", "AC")) == 0
    elif mode == "humidity_priority":
        assert min(t) >= 24.0 - 0.001
        assert max(t) < 25.9
        assert max(rh) < 60.05
        assert max(s("aircon_reheat", "AC")) == 0
    else:
        assert np.allclose(t, 26.0, atol=0.01)
        assert max(rh) < 60.05
        assert min(s("aircon_reheat", "AC")) > 0
        assert np.all(s("aircon_power", "AC") >= s("aircon_reheat", "AC"))
    assert np.all(s("humidity_x", "AC") <= s("humidity_x", "room") + 1e-7)


def test_supply_must_dry_below_room_target_and_conserve_water():
    raw = case()
    raw["ventilation_branches"][0]["humidity_generation"] = 0.00002
    s = run(raw)
    room = s("humidity_x", "room")
    supply = s("humidity_x", "AC")
    goal = humidity_ratio_from_rh(26.0, 60.0)
    assert np.allclose(room, goal, atol=2e-7)
    assert max(supply) < goal - 0.001
    previous = np.r_[0.018, room[:-1]]
    # C dx/dt = outdoor + recirculation + generation, independently of control.
    residual = 1.2 * 40 * (room - previous) / 3600 - (
        1.2 * 0.05 * (0.018 - room) + 1.2 * 0.1 * (supply - room) + 0.00002)
    assert max(abs(residual)) < 2e-7


def test_overcooling_limit_reports_unmet_humidity():
    raw = case("humidity_priority")
    raw["aircon"][0]["max_overcooling"] = 0.2
    s = run(raw)
    assert min(s("thermal_temperature", "room")) >= 25.8 - 0.001
    assert max(s("aircon_humidity_unmet", "AC")) > 1.0


def test_capacity_shortfall_is_not_numerical_failure():
    raw = case()
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = 0.4
    s = run(raw)
    assert max(s("aircon_sensible_heat", "AC") + s("aircon_latent_heat", "AC")) <= 400.01
    assert min(s("aircon_humidity_unmet", "AC")) > 1.0


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_dry_room_does_not_humidify(mode):
    s = run(case(mode, x=0.006))
    assert max(s("aircon_latent_heat", "AC")) == 0
    assert max(s("aircon_reheat", "AC")) == 0
    assert max(s("humidity_x", "room")) <= 0.0060001


@pytest.mark.parametrize("field,value", [("control_mode", "bogus"), ("max_overcooling", -1),
                                        ("min_coil_temperature", -5), ("reheat_cop", 0)])
def test_invalid_controls_rejected(field, value):
    raw = case()
    raw["aircon"][0][field] = value
    with pytest.raises((ValueError, TypeError, ValidationError)):
        build_config(raw, output_path=None, add_surface=False, add_capacity=False)


def test_strict_api_accepts_control_fields_and_builder_preserves_them():
    raw = case()
    RawSimConfig.model_validate(deepcopy(raw))
    built = build_config(raw, output_path=None, add_surface=False, add_capacity=False)
    ac = next(n for n in built["nodes"] if n["type"] == "aircon")
    assert ac["control_mode"] == "temperature_humidity"
    assert ac["max_overcooling"] == 2.0


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
@pytest.mark.parametrize("moist", [False, True])
def test_remote_room_and_discharge_plenum(mode, moist):
    raw = case(mode, moist=moist)
    raw["nodes"].extend([
        {"key": "return", "t": 26.0, "x": 0.018, "calc_t": True, "calc_x": True, "v": 10.0},
        {"key": "supply", "t": 26.0, "x": 0.018, "calc_t": True, "calc_x": True, "v": 0.0},
    ])
    raw["aircon"][0].update({"in": "return", "out": "supply"})
    raw["ventilation_branches"].extend([
        {"key": "room->return", "vol": 0.1}, {"key": "supply->room", "vol": 0.1}])
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")) < 60.1
    assert min(s("thermal_temperature", "room")) >= 24.0 - 0.001
    assert max(abs(s("humidity_x", "supply") - s("humidity_x", "AC"))) < 1e-8


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_off_and_dry_wet_schedule(mode):
    raw = case(mode)
    raw["aircon"][0]["mode"] = ["COOLING", "OFF", "COOLING", "COOLING", "OFF", "COOLING"]
    raw["nodes"][0]["x"] = [0.018, 0.006, 0.006, 0.018, 0.018, 0.018]
    raw["aircon"][0]["pre_rh"] = [60.0, 60.0, 60.0, 55.0, 60.0, 60.0]
    s = run(raw)
    for i in (1, 4):
        assert s("aircon_latent_heat", "AC")[i] == 0
        assert s("aircon_reheat", "AC")[i] == 0
        assert s("aircon_humidity_unmet", "AC")[i] == 0
    assert np.all(np.isfinite(s("aircon_room_rh", "AC")))


def test_cold_room_does_not_heat_for_humidity_priority():
    s = run(case("humidity_priority", outside_t=22.0))
    assert max(s("thermal_temperature", "room")) <= 22.01
    assert max(s("aircon_latent_heat", "AC")) == 0
    assert min(s("aircon_humidity_unmet", "AC")) > 0


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_minimum_coil_temperature_limits_sensible_cooling(mode):
    raw = case(mode, outside_t=38.0)
    s = run(raw)
    assert min(s("thermal_temperature", "AC")) >= 5.0 - 0.002
    assert max(s("aircon_temperature_unmet", "AC")) > 0.1


def test_reheat_cop_changes_power_without_changing_water_or_heat():
    raw = case()
    s1 = run(raw)
    raw["aircon"][0]["reheat_cop"] = 2.0
    s2 = run(raw)
    assert np.allclose(s1("humidity_x", "room"), s2("humidity_x", "room"))
    assert np.allclose(s1("aircon_power", "AC") - s2("aircon_power", "AC"),
                       s1("aircon_reheat", "AC") / 2, atol=0.001)


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_duct_humidity_only_flow_remains_positive(mode):
    raw = case(mode)
    ac = raw["aircon"][0]
    ac["model"] = "DUCT_CENTRAL"
    ac["ac_spec"]["V_inner"]["cooling"]["dsgn"] = 0.1
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")) < 60.1
    assert min(s("aircon_latent_heat", "AC")) > 0

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
    if mode != "temperature_priority":
        ac["ac_spec"]["latent_method"] = "ideal_coil"
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
        assert max(s("aircon_temperature_unmet", "AC")) == 0
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


def test_prescribed_moisture_sink_does_not_break_boundary_response():
    raw = case()
    raw["nodes"][1]["v"] = 0.0
    raw["nodes"].append({"key": "return", "t": 28.0, "x": 0.018})
    raw["aircon"][0]["in"] = "return"
    raw["ventilation_branches"] = [
        {"key": "room->return", "vol": 0.1},
        {"key": "outside->room", "vol": 0.0, "humidity_generation": -0.00001},
    ]
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")) < 60.05
    assert min(s("humidity_x", "room")) > 0.0


def test_simultaneous_mode_uses_reheat_for_a_cold_dry_room():
    s = run(case("temperature_humidity", outside_t=22.0, x=0.006))
    assert np.allclose(s("thermal_temperature", "room"), 26.0, atol=0.001)
    assert min(s("aircon_reheat", "AC")) > 0
    assert max(s("aircon_latent_heat", "AC")) == 0
    assert np.allclose(s("aircon_power", "AC"), s("aircon_reheat", "AC"), atol=0.001)


@pytest.mark.parametrize("method", ["rh95", "bf", "coil_aoaf"])
@pytest.mark.parametrize("moist", [False, True])
def test_humidity_priority_uses_selected_coil(method, moist):
    raw = case("humidity_priority", moist=moist)
    raw["aircon"][0]["ac_spec"].update(latent_method=method, Ao=2.0)
    raw["aircon"][0]["max_overcooling"] = 12.0
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")) <= 60.05
    assert max(s("aircon_reheat", "AC")) == 0
    assert min(s("aircon_latent_heat", "AC")) > 0
    assert min(s("aircon_sensible_heat", "AC")) > 0
    assert min(s("aircon_power", "AC")) > 0
    # It must not silently substitute the ideal saturated coil.
    raw["aircon"][0]["ac_spec"]["latent_method"] = "ideal_coil"
    ideal = run(raw)
    assert np.max(abs(s("thermal_temperature", "room") - ideal("thermal_temperature", "room"))) > 0.01


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_remote_humidistat_with_supply_moisture_source(mode):
    raw = case(mode)
    raw["nodes"].extend([
        {"key": "return", "t": 26.0, "x": 0.018, "calc_t": True, "calc_x": True, "v": 10.0},
        {"key": "supply", "t": 26.0, "x": 0.018, "calc_t": True, "calc_x": True, "v": 0.0},
    ])
    raw["aircon"][0].update({"in": "return", "out": "supply", "max_overcooling": 12.0})
    raw["ventilation_branches"].extend([
        {"key": "room->return", "vol": 0.1},
        {"key": "supply->room", "vol": 0.1, "humidity_generation": 0.00003}])
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")) < 60.05
    room, supply = s("humidity_x", "room"), s("humidity_x", "supply")
    previous = np.r_[0.018, room[:-1]]
    balance = 1.2*40*(room-previous)/3600 - (1.2*.05*(.018-room)+1.2*.1*(supply-room)+.00003)
    assert max(abs(balance)) < 2e-7


def connected_two_ac_case(mode, count=6):
    raw = case(mode, count=count)
    raw["aircon"][0]["max_overcooling"] = 12.0
    room = deepcopy(raw["nodes"][1])
    room["key"] = "room2"
    raw["nodes"].append(room)
    raw["thermal_branches"].append({"key": "outside->room2", "conductance": 1000.0})
    ac = deepcopy(raw["aircon"][0])
    ac.update({"key": "AC2", "in": "room2", "out": "room2", "set": "room2", "pre_rh": 58.0})
    raw["aircon"].append(ac)
    raw["ventilation_branches"].extend([
        {"key": "outside->room2", "vol": 0.05, "humidity_generation": 0.00002},
        {"key": "room2->outside", "vol": 0.05},
        {"key": "room->room2", "vol": 0.03},
        {"key": "room2->room", "vol": 0.03}])
    return raw


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_two_interacting_humidistats_match_targets_and_balance(mode):
    s = run(connected_two_ac_case(mode))
    for room, unit, other, goal, generation in (("room", "AC", "room2", 60.0, 0.0),
                                               ("room2", "AC2", "room", 58.0, .00002)):
        assert max(abs(s("aircon_room_rh", unit)-goal)) < .05
        x = s("humidity_x", room)
        previous = np.r_[.018, x[:-1]]
        balance = 1.2*40*(x-previous)/3600 - (
            1.2*.05*(.018-x)+1.2*.1*(s("humidity_x", unit)-x)+1.2*.03*(s("humidity_x", other)-x)+generation)
        assert max(abs(balance)) < 2e-7


def test_two_aircons_on_same_set_room_are_explicitly_rejected():
    raw = case()
    second = deepcopy(raw["aircon"][0])
    second["key"] = "AC2"
    raw["aircon"].append(second)
    from app import solver_runner
    built = build_config(raw, output_path=None, add_surface=False, add_capacity=False)
    output = solver_runner.run_solver(built)
    assert output["status"] == "error"
    assert "multiple aircons control the same set_node" in output["error"]


@pytest.mark.parametrize("method", ["bf", "coil_aoaf"])
def test_selected_coil_temperature_limit_reports_unmet(method):
    raw = case("humidity_priority")
    ac = raw["aircon"][0]
    ac.update(max_overcooling=12.0, min_coil_temperature=18.0)
    ac["ac_spec"].update(latent_method=method, Ao=.5, bf=.8)
    s = run(raw)
    assert min(s("aircon_humidity_unmet", "AC")) > 1
    assert min(s("thermal_temperature", "AC")) >= 18.0 - .002


def test_naturally_cool_room_is_not_cooling_temperature_unmet():
    s = run(case("temperature_priority", outside_t=22.0))
    assert max(s("aircon_temperature_unmet", "AC")) == 0


@pytest.mark.parametrize("target", [56.0, 58.0])
def test_sensible_saturation_on_one_unit_does_not_overdry_the_other(target):
    raw = connected_two_ac_case("temperature_humidity")
    raw["nodes"][0].update(t=27.3867607, x=[.016529355]*6)
    raw["aircon"][1]["pre_rh"] = target
    s = run(raw)
    assert max(s("aircon_room_rh", "AC")[1:]) < 59.0  # sensible load forces extra condensation
    if target == 56.0:
        assert max(abs(s("aircon_room_rh", "AC2")-target)) < .01
    else:
        # Both outlets must already be saturated for the sensible load. Extra
        # dehumidification/reheat would needlessly push RH even lower.
        assert max(s("aircon_reheat", "AC2")[1:]) < .01


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
@pytest.mark.parametrize("capacity_kw", [.4, 1.0, 2.0])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("moist", [False, True])
def test_one_capacity_limited_humidistat_keeps_other_room_on_target(mode, capacity_kw, reverse, moist):
    raw = connected_two_ac_case(mode)
    raw["simulation"]["coupling"]["moist_enthalpy_enabled"] = moist
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = capacity_kw
    raw["aircon"][1]["min_coil_temperature"] = 0.0  # isolate equipment capacity from the coil floor
    if reverse:
        raw["aircon"].reverse()
    s = run(raw)
    # The weak unit may miss RH, but its feasible boundary must be reflected
    # in the other room, whose coil and 20 kW capacity both have headroom.
    assert max(abs(s("aircon_room_rh", "AC2") - 58.0)) < .05
    assert min(s("aircon_humidity_unmet", "AC")) > .1
    q = s("aircon_sensible_heat", "AC") + s("aircon_latent_heat", "AC")
    assert max(q) <= capacity_kw*1000*1.001 + 1.01
    if mode == "humidity_priority":
        # With unmet RH and no coil/overcooling boundary, use available capacity.
        assert min(q) >= capacity_kw*1000*.999 - 1.01
    for room, ac, other, generation in [("room","AC","room2",0.), ("room2","AC2","room",.00002)]:
        x = s("humidity_x", room)
        residual = 1.2*40*(x-np.r_[.018,x[:-1]])/3600 - (
            1.2*.05*(.018-x)+1.2*.1*(s("humidity_x",ac)-x)+1.2*.03*(s("humidity_x",other)-x)+generation)
        assert max(abs(residual)) < 2e-7


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_capacity_limited_humidistat_can_stop_and_restart(mode):
    raw = connected_two_ac_case(mode)
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = 1.0
    raw["aircon"][1]["min_coil_temperature"] = 0.0
    raw["aircon"][0]["mode"] = ["COOLING", "OFF", "COOLING", "OFF", "COOLING", "COOLING"]
    s = run(raw)
    assert max(abs(s("aircon_room_rh", "AC2")-58.0)) < .05
    for field in ("aircon_sensible_heat", "aircon_latent_heat", "aircon_reheat"):
        assert max(abs(s(field,"AC")[[1,3]])) == 0.0


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_latent_capacity_constraint_releases_when_outdoor_air_dries(mode):
    raw = connected_two_ac_case(mode)
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = 1.0
    raw["aircon"][1]["min_coil_temperature"] = 0.0
    raw["nodes"][0]["x"] = [.018,.006,.018,.006,.018,.006]
    s = run(raw)
    assert max(abs(s("aircon_room_rh", "AC2")[[0,2,4]]-58.0)) < .05
    for ac in ("AC", "AC2"):
        assert max(s("aircon_humidity_unmet",ac)[[1,3,5]]) == 0.0
        assert max(s("aircon_latent_heat",ac)[[1,3,5]]) == 0.0
        assert max(s("aircon_reheat",ac)[[1,3,5]]) == 0.0


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_three_interacting_units_with_one_capacity_limit(mode):
    raw = connected_two_ac_case(mode)
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = 1.0
    raw["aircon"][1]["min_coil_temperature"] = 0.0
    room = deepcopy(raw["nodes"][-1])
    room["key"] = "room3"
    raw["nodes"].append(room)
    raw["thermal_branches"].append({"key":"outside->room3", "conductance":1000.0})
    third = deepcopy(raw["aircon"][1])
    third.update({"key":"AC3","in":"room3","out":"room3","set":"room3","pre_rh":56.0})
    raw["aircon"].append(third)
    raw["ventilation_branches"].extend([
        {"key":"outside->room3", "vol":.05}, {"key":"room3->outside", "vol":.05},
        {"key":"room2->room3", "vol":.02}, {"key":"room3->room2", "vol":.02}])
    s = run(raw)
    for ac, goal in [("AC2",58.0),("AC3",56.0)]:
        assert max(abs(s("aircon_room_rh",ac)-goal)) < .05
    assert min(s("aircon_humidity_unmet","AC")) > .1


@pytest.mark.parametrize("method", ["rh95", "bf", "coil_aoaf"])
@pytest.mark.parametrize("moist", [False, True])
def test_capacity_coupling_with_physical_latent_methods(method, moist):
    raw = connected_two_ac_case("humidity_priority")
    raw["simulation"]["coupling"]["moist_enthalpy_enabled"] = moist
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = 1.0
    raw["aircon"][1]["vol"] = .15  # retain coil-temperature headroom even for BF
    for ac in raw["aircon"]:
        ac["ac_spec"]["latent_method"] = method
        ac["min_coil_temperature"] = 0.0
    s = run(raw)
    assert max(abs(s("aircon_room_rh", "AC2")-58.0)) < .05
    q = s("aircon_sensible_heat", "AC") + s("aircon_latent_heat", "AC")
    assert min(q) >= 997.99 and max(q) <= 1002.01


@pytest.mark.parametrize("moist", [False, True])
def test_both_humidistats_capacity_limited(moist):
    raw = connected_two_ac_case("humidity_priority")
    raw["simulation"]["coupling"]["moist_enthalpy_enabled"] = moist
    for ac in raw["aircon"]:
        ac["ac_spec"]["Q"]["cooling"]["max"] = 1.0
    s = run(raw)
    for ac in ("AC", "AC2"):
        assert min(s("aircon_humidity_unmet", ac)) > .1
        q = s("aircon_sensible_heat", ac) + s("aircon_latent_heat", ac)
        assert min(q) >= 997.99 and max(q) <= 1002.01


@pytest.mark.parametrize("capacity_kw", [.4, 1.0, 2.0])
@pytest.mark.parametrize("moist", [False, True])
def test_equipment_capacity_and_other_units_coil_floor(capacity_kw, moist):
    raw = connected_two_ac_case("humidity_priority")
    raw["simulation"]["coupling"]["moist_enthalpy_enabled"] = moist
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = capacity_kw
    s = run(raw)
    # Preserve the original 5 C coil limit: early hours can be infeasible,
    # but a constrained unit must not prevent the other root from converging.
    supply = s("thermal_temperature", "AC2")
    assert min(supply) >= 4.999
    # The coil-derived watt limit reserves its acceptance band inside the
    # temperature floor; that band can leave about 0.03 K of apparent room.
    free = supply > 5.05
    assert free.any()
    assert max(abs(s("aircon_room_rh", "AC2")[free]-58.0)) < .05
    q = s("aircon_sensible_heat", "AC") + s("aircon_latent_heat", "AC")
    assert min(q) >= capacity_kw*1000*.999-1.01
    assert max(q) <= capacity_kw*1000*1.001+1.01

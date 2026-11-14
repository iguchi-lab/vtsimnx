"""Acceptance is both physically balanced and consistent with control intent."""
import numpy as np
import pytest

from app.builder import build_config
from app.builder.validate import ValidationError
import app.solver_runner as sr
from tests_py.test_aircon_saturation import isolated_solver, run  # noqa: F401
from tests_py.physics.test_coil_latent_continuity import _raw_coil_case


def cooling_case(model="RAC", limit=7.0):
    raw = _raw_coil_case([0.018] * 3, duct=True)
    raw["aircon"][0]["model"] = model
    raw["aircon"][0]["ac_spec"]["Q"]["cooling"]["max"] = limit
    return raw


@pytest.mark.parametrize("model", ["RAC", "DUCT_CENTRAL"])
def test_capacity_search_retains_a_feasible_requested_temperature(model):
    reference = run(cooling_case(model, 20.0))
    assert max(reference("aircon_sensible_heat", "AC") +
               reference("aircon_latent_heat", "AC")) < 7000
    limited = run(cooling_case(model, 7.0))
    np.testing.assert_allclose(limited("thermal_temperature", "room"), 27.5, atol=0.002)
    assert max(limited("aircon_sensible_heat", "AC") +
               limited("aircon_latent_heat", "AC")) <= 7008


def test_true_capacity_shortfall_remains_bounded():
    s = run(cooling_case(limit=4.0))
    assert min(s("thermal_temperature", "room")) > 27.5
    assert max(s("aircon_sensible_heat", "AC") + s("aircon_latent_heat", "AC")) <= 4005


@pytest.mark.parametrize("field,value", [
    ("latent_method", "misspelled"), ("bf", -0.1), ("bf", 1.0), ("bf", 0.995),
    ("Af", -0.3), ("Ao", 0.0),
])
def test_explicit_invalid_latent_inputs_are_rejected(field, value):
    raw = cooling_case()
    raw["aircon"][0]["ac_spec"][field] = value
    with pytest.raises((ValidationError, ValueError)):
        build_config(raw, output_path=None, add_surface=False, add_capacity=False)


def test_rh100_is_an_explicit_alias_for_saturation_only():
    raw = cooling_case()
    raw["aircon"][0]["ac_spec"]["latent_method"] = "none"
    a = run(raw)
    raw["aircon"][0]["ac_spec"]["latent_method"] = "rh100"
    b = run(raw)
    for field in ("humidity_x", "aircon_sensible_heat", "aircon_latent_heat", "aircon_power"):
        np.testing.assert_array_equal(a(field, "AC"), b(field, "AC"))


def test_missing_duct_spec_rejected_before_solve():
    raw = cooling_case("DUCT_CENTRAL")
    del raw["aircon"][0]["ac_spec"]["P_fan"]
    with pytest.raises((ValidationError, ValueError), match="P_fan"):
        build_config(raw, output_path=None, add_surface=False, add_capacity=False)


def test_direct_solver_does_not_turn_missing_power_spec_into_zero_success():
    built = build_config(cooling_case("DUCT_CENTRAL"), output_path=None,
                         add_surface=False, add_capacity=False)
    ac = next(n for n in built["nodes"] if n["type"] == "aircon")
    del ac["ac_spec"]["P_fan"]
    result = sr.run_solver(built)
    assert result["status"] == "error", result


@pytest.mark.parametrize("mode", ["humidity_priority", "temperature_humidity"])
def test_unsupported_humidistat_coil_combination_is_not_silently_ignored(mode):
    raw = cooling_case()
    raw["aircon"][0].update(control_mode=mode, pre_rh=60.0)
    with pytest.raises((ValidationError, ValueError), match="latent_method"):
        build_config(raw, output_path=None, add_surface=False, add_capacity=False)


def fan_case(off_flow, initial_outdoor=32.0):
    raw = cooling_case("DUCT_CENTRAL", 20.0)
    raw["nodes"][0]["t"] = [initial_outdoor, 24.0, 24.0]
    raw["nodes"][1].update(calc_p=True, p=0.0)
    raw["simulation"]["calc_flag"]["p"] = True
    ac = raw["aircon"][0]
    ac.pop("vol")
    ac.update(p_max=440.0, p1=200.0, q1=0.3777777778, q_max=0.5222222222,
              mode=["COOLING", "OFF", "OFF"], vol_zero=off_flow)
    return raw


@pytest.mark.parametrize("off_flow", [0.0, 0.04])
def test_stopped_compressor_has_explicit_fan_state_and_power(off_flow):
    a = run(fan_case(off_flow))
    b = run(fan_case(off_flow, 30.0))
    for s in (a, b):
        assert max(s("aircon_sensible_heat", "AC")[1:]) == 0
        assert max(s("aircon_latent_heat", "AC")[1:]) == 0
        flow = s("vent_flow_rate", "room->AC")[1:]
        power = s("aircon_power", "AC")[1:]
        if off_flow == 0:
            assert max(abs(flow)) < 1e-4
            assert max(power) == 0
        else:
            assert min(flow) > 0.01
            assert min(power) > 1.0
    np.testing.assert_allclose(a("vent_flow_rate", "room->AC")[1:],
                               b("vent_flow_rate", "room->AC")[1:], atol=1e-4)


def test_ideal_cooling_still_accounts_for_explicit_reheat():
    from tests_py.test_room_humidity_control import case
    raw = case()
    raw["aircon"][0].update(model="IDEAL", reheat_cop=2.0)
    s = run(raw)
    assert min(s("aircon_reheat", "AC")) > 0
    np.testing.assert_allclose(s("aircon_power", "AC"), s("aircon_reheat", "AC") / 2)


def test_heating_only_off_ventilation_uses_heating_fan_spec():
    from tests_py.test_heating_fan_coupling import heating_fan_case
    raw = heating_fan_case(restart=True)
    raw["aircon"][0]["vol_zero"] = 0.04
    s = run(raw)
    assert s("aircon_sensible_heat", "AC")[1] == 0
    assert s("aircon_power", "AC")[1] > 1
    assert s("vent_flow_rate", "room->AC")[1] > 0.01

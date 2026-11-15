"""Humidity-only starts/stops must work with actual PQ-network airflow."""
import numpy as np
import pytest

from tests_py.test_aircon_saturation import isolated_solver, run  # noqa: F401
from tests_py.physics.test_coil_latent_continuity import _raw_coil_case
from vtsimnx.archenv.archenv import saturation_vapor_pressure_pa, vapor_pressure_from_humidity_ratio_pa


def fan_case(method="rh95", control="humidity_priority"):
    wet_dry = [0.018] * 3 + [0.006] * 3 + [0.018] * 4
    raw = _raw_coil_case(wet_dry, duct=True)
    raw["simulation"]["calc_flag"]["p"] = True
    raw["nodes"][0]["t"] = 27.0
    raw["nodes"][1].update(t=27.0, x=0.018, calc_p=True, p=0.0)
    raw["thermal_branches"][0]["conductance"] = 500.0
    for branch in raw["ventilation_branches"]:
        branch["vol"] = 0.02
    raw["nodes"].extend([
        {"key": "return", "t": 27.0, "x": 0.018, "calc_t": True,
         "calc_x": True, "calc_p": True, "v": 20.0},
        {"key": "supply", "t": 27.0, "x": 0.018, "calc_t": True,
         "calc_x": True, "calc_p": True, "v": 0.0},
    ])
    ac = raw["aircon"][0]
    ac.update({"in": "return", "out": "supply", "pre_temp": 27.0, "pre_rh": 60.0,
               "control_mode": control, "max_overcooling": 2.0,
               "mode": ["COOLING"] * 9 + ["OFF"], "p_max": 440.0, "p1": 200.0,
               "q1": 0.3777777778, "q_max": 0.5222222222})
    ac.pop("vol")
    ac["ac_spec"].update(latent_method="bf" if method == "bf_fixed" else method,
                         bf=0.1, bf_flow_correction=method != "bf_fixed")
    raw["ventilation_branches"].extend([
        {"key": "supply->room", "alpha": 0.65, "area": 1.0},
        {"key": "room->return", "alpha": 0.65, "area": 1.0},
    ])
    return raw


@pytest.mark.parametrize("method", ["rh95", "bf_fixed", "bf", "coil_aoaf", "none"])
def test_wet_dry_restart_with_zero_temperature_demand(method):
    raw = fan_case(method)
    s = run(raw)
    wet = np.array([0, 1, 2, 6, 7, 8])
    rh = s("aircon_room_rh", "AC")
    qs = s("aircon_sensible_heat", "AC")
    ql = s("aircon_latent_heat", "AC")
    power = s("aircon_power", "AC")
    t = s("thermal_temperature", "room")
    flow = s("vent_flow_rate", "return->AC")
    # Outside and requested room temperatures are identical. Humidity alone
    # must start/restart cooling, with real sensible cooling and electricity.
    assert np.all(ql[wet] > 1)
    assert np.all(qs[wet] > 1)
    assert np.all(power[wet] > 0)
    assert np.max(rh[wet]) <= 60.15
    assert np.min(t[wet]) >= 25.0 - 0.001
    assert np.max(t[wet]) < 27.0
    assert np.max(flow[wet]) < 0.4  # full PQ speed delivers about 0.52 m3/s
    # After dry air removes the demand, no sensible load may keep the unit on.
    assert ql[5] == 0 and qs[5] == 0 and power[5] == 0
    assert abs(flow[5]) < 1e-4
    # Explicit OFF wins even when humidity returns.
    assert ql[9] == 0 and qs[9] == 0 and power[9] == 0
    assert abs(flow[9]) < 1e-4
    for key in ("AC", "supply"):
        outlet_rh = 100 * vapor_pressure_from_humidity_ratio_pa(s("humidity_x", key)) / \
            saturation_vapor_pressure_pa(s("thermal_temperature", key))
        assert np.max(outlet_rh) <= 100.0
    # Independent room water ledger, with both actual fan branches and fixed
    # outdoor exchange. Inventory is always the previous accepted hour.
    x = s("humidity_x", "room")
    previous = np.r_[0.018, x[:-1]]
    residual = 1.2 * 40 * (x - previous) / 3600 - 1.2 * (
        0.02 * (np.asarray(raw["nodes"][0]["x"]) - x) +
        s("vent_flow_rate", "supply->room") * s("humidity_x", "supply") -
        s("vent_flow_rate", "room->return") * x)
    assert np.max(abs(residual)) < 3e-8


def test_temperature_priority_does_not_start_for_humidity():
    s = run(fan_case(control="temperature_priority"))
    assert np.max(s("aircon_latent_heat", "AC")) == 0
    assert np.max(s("aircon_sensible_heat", "AC")) < 0.01
    assert np.max(s("aircon_power", "AC")) == 0

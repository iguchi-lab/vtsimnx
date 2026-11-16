"""Accepted coil states respect the wet-surface bounds at network airflow."""
import numpy as np
import pytest

from tests_py.physics.test_coil_latent_continuity import _raw_coil_case
from tests_py.test_aircon_saturation import isolated_solver, run  # noqa: F401
from vtsimnx.archenv.archenv import humidity_ratio_from_rh


@pytest.mark.parametrize("method", ["bf", "coil_aoaf"])
@pytest.mark.parametrize("control", ["temperature_priority", "humidity_priority"])
def test_coil_surface_floor_applies_to_both_controls(method, control, tmp_path):
    raw = _raw_coil_case([.018] * 4, duct=True)
    ac = raw["aircon"][0]
    ac.update(control_mode=control, min_coil_temperature=20.0, pre_rh=50.0)
    ac["ac_spec"].update(latent_method=method, bf=.8, Ao=.5)
    s = run(raw)
    tin = s("thermal_temperature", "room")
    tout = s("thermal_temperature", "AC")
    xin = s("humidity_x", "room")
    xout = s("humidity_x", "AC")
    flow = np.abs(s("vent_flow_rate", "room->AC"))
    np.testing.assert_allclose(s("aircon_sensible_heat", "AC"),
                               1.2*1006*flow*(tin-tout), atol=.02, rtol=1e-5)
    if method == "bf":
        bf = .8 ** (ac["ac_spec"]["V_inner"]["cooling"]["rtd"] / flow)
        coil = tin - (tin-tout) / (1-bf)
        assert np.max(coil-tout) <= .003
        xsat = humidity_ratio_from_rh(coil, 100)
        assert np.min(xout-np.minimum(xin, xsat)) >= -2e-7
    else:
        # Independent reconstruction of report steps 4-8 with the documented
        # dry-boundary cap. No BF/NTU identity is assumed for this model.
        hs = s("aircon_sensible_heat", "AC")
        te = tin - hs / (1.2 * 1006 * flow)
        xe = np.minimum(xin, humidity_ratio_from_rh(te, 100))
        star = (xin + xe) / 2
        coefficient = np.maximum(0, .037 * np.log(flow / .133) + .0637)
        coil = (tin + te) / 2 - hs / (coefficient * (1006 + 1846 * star) * .5)
        xd = humidity_ratio_from_rh(coil, 100)
        hr = np.where(xin > xd, np.maximum(0, (2500800 + 1846 * coil) * coefficient * (star-xd) * .5), 0)
        expected = np.clip(xin - hr / ((2500800-2366.8*tout)*1.2*flow),
                           0, np.minimum(xin, humidity_ratio_from_rh(tout, 100)))
        np.testing.assert_allclose(xout, expected, atol=2e-7, rtol=0)
    assert np.min(coil) >= 20 - .003
    assert np.min(s("aircon_temperature_unmet", "AC")) > .1
    log = next(tmp_path.rglob("solver.log")).read_text()
    # Speculative points never export a timestep or advance its inventory.
    assert log.count("最終状態整合") == 4


def test_bf_uses_network_flow_and_reference_default():
    raw = _raw_coil_case([.018] * 3, duct=True)
    ac = raw["aircon"][0]
    ac["model"] = "RAC"  # fixed circuit airflow; DUCT_CENTRAL modulates the network fan
    for mode in ("cooling", "heating"):
        ac["ac_spec"]["Q"][mode]["max"] = ac["ac_spec"]["Q"][mode]["rtd"]
    ac["ac_spec"].update(latent_method="bf", bf_reference_flow_rate=.6)
    default = run(raw)
    ac["ac_spec"].update(bf=.1)
    explicit = run(raw)
    for field in ("humidity_x", "aircon_latent_heat", "aircon_power"):
        np.testing.assert_array_equal(default(field, "AC"), explicit(field, "AC"))
    flow = np.abs(explicit("vent_flow_rate", "room->AC"))
    np.testing.assert_allclose(flow, .3, atol=1e-7)
    tin, tout = explicit("thermal_temperature", "room"), explicit("thermal_temperature", "AC")
    xin, xout = explicit("humidity_x", "room"), explicit("humidity_x", "AC")
    bf = .1 ** (.6 / flow)
    coil = tin - (tin-tout) / (1-bf)
    expected = np.minimum(xin, bf*xin+(1-bf)*humidity_ratio_from_rh(coil, 100))
    expected = np.minimum(expected, humidity_ratio_from_rh(tout, 100))
    np.testing.assert_allclose(xout, expected, atol=2e-7, rtol=0)

from copy import deepcopy

import pytest

from app.builder import build_config_with_warnings
from app.builder.validate import ValidationError
from app.schemas.config import UnknownFieldError, prepare_raw_config
from app.solver_runner import force_log_verbosity


def raw():
    return {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00",
                "end": "2025-01-01T01:00:00",
                "length": 2,
                "timestep": 3600,
            }
        },
        "nodes": [{"key": "R", "t": 20}],
        "thermal_branches": [],
        "ventilation_branches": [],
    }


def test_controls_survive_api_and_direct_builder():
    cfg = raw()
    controls = {
        "iteration": {"max_inner": 7, "max_coupling": 8, "max_aircon_control": 9},
        "coupling": {
            "moisture_enabled": False,
            "humidity_relaxation": 0.7,
            "temperature_relaxation": 0.5,
            "latent_coupling_mode": "disabled",
            "moist_enthalpy_enabled": False,
        },
        "log": {"verbosity": 0},
        "max_aircon_control_iteration": 11,
    }
    cfg["simulation"].update(controls)
    for route in (
        lambda c: c,
        lambda c: prepare_raw_config(c, unknown_keys="error")[0],
    ):
        built, warnings = build_config_with_warnings(route(deepcopy(cfg)))
        assert warnings == []
        for name, value in controls.items():
            assert built["simulation"][name] == value
        force_log_verbosity(built, debug=False, debug_verbosity=2)
        assert built["simulation"]["log"]["verbosity"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("humidity_relaxation", 0),
        ("temperature_relaxation", 0),
        ("latent_relaxation", 1.1),
        ("humidity_solver_tolerance", -1),
    ],
)
def test_direct_builder_rejects_invalid_coupling(field, value):
    cfg = raw()
    cfg["simulation"]["coupling"] = {field: value}
    with pytest.raises(ValidationError):
        build_config_with_warnings(cfg)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_iteration_limits_require_positive_integers(value):
    cfg = raw()
    cfg["simulation"]["iteration"] = {"max_coupling": value}
    with pytest.raises(ValidationError):
        build_config_with_warnings(cfg)


def test_nested_unknown_controls_warn_or_reject():
    cfg = raw()
    cfg["simulation"]["coupling"] = {"moisture_enable_typo": True}
    built, warnings = build_config_with_warnings(cfg)
    assert built["simulation"]["coupling"] == {}
    assert any("moisture_enable_typo" in w for w in warnings)
    with pytest.raises(UnknownFieldError):
        build_config_with_warnings(cfg, unknown_keys="error")


def test_debug_can_raise_explicit_log_level():
    cfg = raw()
    cfg["simulation"]["log"] = {"verbosity": 0}
    force_log_verbosity(cfg, debug=True, debug_verbosity=2)
    assert cfg["simulation"]["log"]["verbosity"] == 2


def test_hrv_schema_preserves_ports_fans_and_disable_option():
    cfg = raw()
    cfg["nodes"].append({"key": "OA", "t": 0})
    hrv = {
        "key": "HRV",
        "oa": "OA",
        "out": "R",
        "in": "R",
        "ea": "OA",
        "recovery": "total",
        "eta_t": 0.7,
        "eta_x": 0.6,
        "supply": {"p_max": 100, "p1": 50, "q1": 0.03, "q_max": 0.06},
        "exhaust": {"p_max": 100, "p1": 50, "q1": 0.03, "q_max": 0.06},
    }
    cfg["heat_recovery_vent"] = [hrv]
    cfg["builder"] = {"add_heat_recovery_vent": False}
    prepared, warnings, _ = prepare_raw_config(cfg, unknown_keys="error")
    assert prepared["heat_recovery_vent"] == [hrv]
    assert warnings == []
    built, _ = build_config_with_warnings(prepared)
    assert "HRV" not in {n["key"] for n in built["nodes"]}

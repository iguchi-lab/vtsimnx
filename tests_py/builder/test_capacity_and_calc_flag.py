from app.builder import build_config


def test_thermal_mass_is_converted_to_capacity_node_and_branch_and_removed():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 10, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "N1", "thermal_mass": 100.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=True)

    n1 = next(n for n in out["nodes"] if n["key"] == "N1")
    assert "thermal_mass" not in n1

    assert any(n["key"] == "N1_c" and n.get("type") == "capacity" for n in out["nodes"])
    tb = next(b for b in out["thermal_branches"] if b["key"] == "N1_c->N1")
    assert tb["subtype"] == "capacity"
    assert tb["conductance"] == 100.0 / 10


def test_calc_flag_is_auto_set_from_node_calc_fields():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "N1", "calc_t": True}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=False)
    flags = out["simulation"]["calc_flag"]
    assert flags["t"] is True
    assert flags["p"] is False
    assert flags["x"] is False
    assert flags["c"] is False



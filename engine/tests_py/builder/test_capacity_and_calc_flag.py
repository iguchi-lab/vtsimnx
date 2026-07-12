from app.builder import build_config


def test_thermal_mass_is_converted_to_capacity_node_and_branch_and_removed():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 10, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "N1", "thermal_mass": 100.0, "t": 23.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=True)

    n1 = next(n for n in out["nodes"] if n["key"] == "N1")
    assert "thermal_mass" not in n1

    n1c = next(n for n in out["nodes"] if n["key"] == "N1_c")
    assert n1c.get("type") == "capacity"
    assert n1c.get("t") == 23.0
    tb = next(b for b in out["thermal_branches"] if b["key"] == "N1_c->N1")
    assert tb["subtype"] == "capacity"
    assert tb["conductance"] == 100.0 / 10


def test_thermal_mass_with_volume_splits_air_capacity():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        # ρ cp V = 1.2*1006*50 ≈ 60360, thermal_mass はそれより大きい
        "nodes": [{"key": "ROOM", "v": 50.0, "thermal_mass": 5.0e5, "t": 20.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=True)
    air_mass = 1.2 * 1006.0 * 50.0
    furniture = 5.0e5 - air_mass

    tb_air = next(b for b in out["thermal_branches"] if b["key"] == "ROOM_air->ROOM")
    assert tb_air["subtype"] == "air_capacity"
    assert abs(tb_air["conductance"] - air_mass / 60.0) < 1e-6

    tb_c = next(b for b in out["thermal_branches"] if b["key"] == "ROOM_c->ROOM")
    assert tb_c["subtype"] == "capacity"
    assert abs(tb_c["conductance"] - furniture / 60.0) < 1e-6


def test_thermal_mass_below_air_capacity_is_error():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        # ρ cp V ≈ 60360 より小さい thermal_mass
        "nodes": [{"key": "ROOM", "v": 50.0, "thermal_mass": 1.0e4, "t": 20.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
    }
    try:
        build_config(raw, add_surface=False, add_aircon=False, add_capacity=True)
        assert False, "expected ValueError for thermal_mass < rho*cp*V"
    except ValueError as e:
        assert "thermal_mass" in str(e)
        assert "空気熱容量" in str(e) or "ρ" in str(e) or "rho" in str(e).lower()


def test_manual_air_capacity_conductance_must_match_volume():
    from app.builder.validate import validate_thermal_config

    sim = {
        "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 1},
        "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        "calc_flag": {"p": False, "t": True, "x": False, "c": False},
    }
    nodes = [{"key": "ROOM", "v": 50.0, "t": 20.0}, {"key": "ROOM_air", "type": "capacity", "ref_node": "ROOM", "t": 20.0}]
    # 意図的に小さい conductance
    branches = [
        {
            "key": "ROOM_air->ROOM",
            "type": "conductance",
            "subtype": "air_capacity",
            "conductance": 1.0,
            "source": "ROOM_air",
            "target": "ROOM",
        }
    ]
    _, result = validate_thermal_config(sim, nodes, branches)
    assert not result.is_valid
    assert any("air_capacity" in e for e in result.errors)

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



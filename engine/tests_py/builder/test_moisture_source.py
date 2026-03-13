from app.builder import build_config


def test_humidity_source_adds_void_to_room_vent_branch_and_sets_calc_x():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 3600, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [
            {"key": "LD", "v": 30.0},  # 体積[m3]（validate側でcalc_x時に v を要求する）
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "humidity_source": [
            {"key": "LD_hum", "room": "LD", "generation_rate": [1.0e-4, 2.0e-4]},
        ],
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=False)

    assert out["simulation"]["calc_flag"]["x"] is True
    ld = next(n for n in out["nodes"] if n["key"] == "LD")
    assert ld.get("calc_x") is True

    b = next(v for v in out["ventilation_branches"] if v["key"] == "void->LD")
    assert b["subtype"] == "internal_humidity"
    assert b["vol"] == 0.0
    assert b["humidity_generation"] == [1.0e-4, 2.0e-4]


def test_humidity_source_legacy_dict_is_supported():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 3600, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "BR", "v": 20.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "humidity_source": {
            "BR_mx": {"set": "BR", "mx": [3.0e-5]},
        },
    }

    out = build_config(raw, add_surface=False, add_aircon=False, add_capacity=False)
    assert out["simulation"]["calc_flag"]["x"] is True
    b = next(v for v in out["ventilation_branches"] if v["key"] == "void->BR")
    assert b["humidity_generation"] == [3.0e-5]



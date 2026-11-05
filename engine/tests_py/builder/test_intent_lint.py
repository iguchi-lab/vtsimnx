from app.builder.intent_lint import lint_raw_config


def _base():
    return {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T01:00:00Z",
                "timestep": 3600,
                "length": 2,
            }
        },
        "nodes": [
            {"key": "room", "t": 20.0, "calc_t": True, "v": 30.0, "thermal_mass": 36216.0},
            {"key": "outside", "t": 5.0},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


def test_fixed_flow_supply_without_exhaust_warns():
    raw = _base()
    raw["ventilation_branches"] = [
        {"key": "outside->room", "type": "fixed_flow", "vol": 0.01},
    ]
    codes = {i["code"] for i in lint_raw_config(raw)}
    assert "fixed_flow_imbalance" in codes


def test_balanced_fixed_flow_does_not_warn():
    raw = _base()
    raw["ventilation_branches"] = [
        {"key": "outside->room", "type": "fixed_flow", "vol": 0.01},
        {"key": "room->outside", "type": "fixed_flow", "vol": 0.01},
    ]
    codes = {i["code"] for i in lint_raw_config(raw)}
    assert "fixed_flow_imbalance" not in codes


def test_calc_t_without_capacity_warns():
    raw = _base()
    raw["nodes"][0].pop("v")
    raw["nodes"][0].pop("thermal_mass")
    raw["ventilation_branches"] = [
        {"key": "outside->room", "type": "fixed_flow", "vol": 0.01},
        {"key": "room->outside", "type": "fixed_flow", "vol": 0.01},
    ]
    issues = lint_raw_config(raw)
    assert any(i["code"] == "missing_room_capacity" for i in issues)
    cap = next(i for i in issues if i["code"] == "missing_room_capacity")
    assert cap["path"][:2] == ["nodes", 0]


def test_humidity_source_without_calc_x_warns():
    raw = _base()
    raw["humidity_source"] = [{"room": "room", "generation_rate": 1e-5}]
    issues = lint_raw_config(raw)
    assert any(i["code"] == "humidity_source_without_calc_x" for i in issues)

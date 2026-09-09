import pytest

from app.builder import build_config
from app.builder.validate import ValidationError


def test_aircon_adds_aircon_node_and_ventilation_branches():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室1"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "aircon": [
            {
                "key": "AC1",
                "set": "室1",
                "outside": "外気",
                "pre_temp": 18.0,
                "model": "dummy",
                "mode": "cool",
            }
        ],
    }

    out = build_config(raw, add_surface=False, add_capacity=False)

    keys = {n["key"] for n in out["nodes"]}
    assert "AC1" in keys

    # aircon 経由で2本の fixed_flow ブランチが追加される（室1->AC1, AC1->室1）
    vkeys = {b["key"] for b in out["ventilation_branches"]}
    assert "室1->AC1" in vkeys
    assert "AC1->室1" in vkeys

    # subtype が aircon であること
    subs = [b.get("subtype") for b in out["ventilation_branches"] if b["key"] in ("室1->AC1", "AC1->室1")]
    assert subs == ["aircon", "aircon"]

    ac1 = next(n for n in out["nodes"] if n["key"] == "AC1")
    assert ac1["pre_temp"] == [18.0, 18.0]

    intake = next(b for b in out["ventilation_branches"] if b["key"] == "室1->AC1")
    supply = next(b for b in out["ventilation_branches"] if b["key"] == "AC1->室1")
    assert intake["type"] == "fixed_flow"
    assert supply["type"] == "fixed_flow"
    assert "vol" in intake and "vol" in supply


def test_aircon_fan_pq_keeps_supply_as_pressure_loss_and_ignores_vol():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": True, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "還気"}, {"key": "吹出"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "aircon": [
            {
                "key": "AC1",
                "set": "室1",
                "in": "還気",
                "out": "吹出",
                "outside": "外気",
                "pre_temp": 22.0,
                "model": "DUCT_CENTRAL",
                "mode": "heat",
                "vol": 1000 / 3600,
                "p_max": 80.0,
                "p1": 40.0,
                "q1": 0.1,
                "q_max": 0.25,
                "area": 0.08,
                "k_total": 2.0,
            }
        ],
    }
    raw["nodes"].append({"key": "室1"})

    out = build_config(raw, add_surface=False, add_capacity=False)
    intake = next(b for b in out["ventilation_branches"] if b["key"] == "還気->AC1")
    supply = next(b for b in out["ventilation_branches"] if b["key"] == "AC1->吹出")
    assert intake["type"] == "fan"
    assert intake["subtype"] == "aircon"
    assert intake["p_max"] == 80.0
    assert intake["q_max"] == 0.25
    assert "vol" not in intake
    assert supply["type"] == "pressure_loss"
    assert supply["area"] == 0.08
    assert supply["k_total"] == 2.0


def test_aircon_pre_rh_is_passed_to_aircon_node():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室1"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "aircon": [
            {
                "key": "AC1",
                "set": "室1",
                "outside": "外気",
                "pre_temp": 24.0,
                "pre_rh": 50.0,
                "model": "RAC",
                "mode": "cool",
            }
        ],
    }

    out = build_config(raw, add_surface=False, add_capacity=False)
    ac1 = next(n for n in out["nodes"] if n["key"] == "AC1")
    assert ac1["pre_rh"] == [50.0, 50.0]


def test_aircon_pre_temp_nan_is_filled_when_mode_is_off():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室1"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "aircon": [
            {
                "key": "AC1",
                "set": "室1",
                "outside": "外気",
                "pre_temp": [float("nan"), 22.0],
                "model": "dummy",
                "mode": ["OFF", "HEATING"],
            }
        ],
    }

    out = build_config(raw, add_surface=False, add_capacity=False)
    ac1 = next(n for n in out["nodes"] if n["key"] == "AC1")
    assert ac1["pre_temp"] == [20.0, 22.0]


def test_aircon_pre_temp_nan_raises_when_mode_is_active():
    raw = {
        "simulation": {
            "index": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室1"}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "aircon": [
            {
                "key": "AC1",
                "set": "室1",
                "outside": "外気",
                "pre_temp": [float("nan"), 22.0],
                "model": "dummy",
                "mode": ["HEATING", "HEATING"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="pre_temp\\[0\\].*NaN/None"):
        build_config(raw, add_surface=False, add_capacity=False)



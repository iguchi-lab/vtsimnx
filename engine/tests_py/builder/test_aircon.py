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



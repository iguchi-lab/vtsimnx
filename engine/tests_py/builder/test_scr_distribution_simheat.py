"""SCR 室内配分: 保存的配分（既定）と端条件。旧・非保存経路は不可。"""
from __future__ import annotations

import pytest

from app.builder.surface_solar import process_glass_solar
from app.builder.surfaces import process_surfaces


def _gains(branches):
    return [b for b in branches if b.get("subtype") == "solar_gain"]


def test_default_floor_and_other_split_and_air():
    # Q_tr=120 W, SCR=0.5 → Q_rad=60, SCC=0.25 → Q_conv=30
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.25,
        "solar": [120.0],
    }
    surfaces = [
        window,
        {"key": "A->床", "part": "floor", "area": 10.0, "u_value": 1.0, "eta": 0.8},
        {"key": "A->壁", "part": "wall", "area": 20.0, "u_value": 1.0, "eta": 0.8},
    ]
    branches = process_glass_solar(window, surfaces, sim_length=1, verbose=False)
    gains = _gains(branches)
    floor = sum(b["heat_generation"][0] for b in gains if b["key"].endswith("_floor_s"))
    wall = sum(b["heat_generation"][0] for b in gains if b["key"].endswith("_wall_s"))
    air = sum(b["heat_generation"][0] for b in gains if b["key"] == "void->A")
    assert floor == pytest.approx(30.0)
    assert wall == pytest.approx(30.0)
    assert air == pytest.approx(30.0)
    assert floor + wall == pytest.approx(60.0)


def test_explicit_simheat_matches_default():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.25,
        "solar": [120.0],
        "scr_distribution": "simheat",
    }
    surfaces = [
        window,
        {"key": "A->床", "part": "floor", "area": 10.0, "u_value": 1.0, "eta": 0.8},
        {"key": "A->壁", "part": "wall", "area": 20.0, "u_value": 1.0, "eta": 0.8},
    ]
    default = process_glass_solar({**window, "scr_distribution": None}, surfaces, 1, verbose=False)
    explicit = process_glass_solar(window, surfaces, 1, verbose=False)
    assert [b["heat_generation"] for b in _gains(default)] == [
        b["heat_generation"] for b in _gains(explicit)
    ]


def test_absorptance_area_weights_within_floor_group():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.0,
        "solar": [120.0],
    }
    surfaces = [
        window,
        {"key": "A->床1", "part": "floor", "area": 10.0, "u_value": 1.0, "eta": 0.5},
        {"key": "A->床2", "part": "floor", "area": 20.0, "u_value": 1.0, "eta": 0.75},
        {"key": "A->壁", "part": "wall", "area": 10.0, "u_value": 1.0, "eta": 0.8},
    ]
    branches = process_glass_solar(window, surfaces, sim_length=1, verbose=False)
    by_key = {b["key"]: b["heat_generation"][0] for b in _gains(branches)}
    assert by_key["void->A-床1_floor_s"] == pytest.approx(7.5)
    assert by_key["void->A-床2_floor_s"] == pytest.approx(22.5)
    assert by_key["void->A-壁_wall_s"] == pytest.approx(30.0)


def test_one_group_only_gets_all_radiation():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.0,
        "solar": [100.0],
    }
    surfaces = [
        window,
        {"key": "A->壁", "part": "wall", "area": 8.0, "u_value": 1.0, "eta": 0.8},
    ]
    branches = process_glass_solar(window, surfaces, sim_length=1, verbose=False)
    assert sum(b["heat_generation"][0] for b in _gains(branches)) == pytest.approx(50.0)


def test_zero_absorptance_falls_back_to_area_ratio():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 1.0,
        "SCC": 0.0,
        "solar": [100.0],
    }
    surfaces = [
        window,
        {"key": "A->床1", "part": "floor", "area": 10.0, "u_value": 1.0, "eta": 0.0},
        {"key": "A->床2", "part": "floor", "area": 30.0, "u_value": 1.0, "eta": 0.0},
    ]
    branches = process_glass_solar(window, surfaces, sim_length=1, verbose=False)
    by_key = {b["key"]: b["heat_generation"][0] for b in _gains(branches)}
    assert by_key["void->A-床1_floor_s"] == pytest.approx(25.0)
    assert by_key["void->A-床2_floor_s"] == pytest.approx(75.0)


def test_all_zero_radiation_skips_surface_allocation():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.0,
        "SCC": 0.25,
        "solar": [100.0],
    }
    branches = process_glass_solar(window, [window], sim_length=1, verbose=False)
    gains = _gains(branches)
    assert len(gains) == 1
    assert gains[0]["key"] == "void->A"
    assert gains[0]["heat_generation"][0] == pytest.approx(25.0)


def test_no_receivers_with_positive_radiation_errors():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.0,
        "solar": [100.0],
    }
    with pytest.raises(ValueError, match="吸収面が必要"):
        process_glass_solar(window, [window], sim_length=1, verbose=False)


def test_multi_timestep_and_windows_conserve():
    windows = [
        {
            "key": "A->外1",
            "part": "glass",
            "area": 2.0,
            "u_value": 5.0,
            "SCR": 0.4,
            "SCC": 0.1,
            "solar": [50.0, 100.0],
        },
        {
            "key": "A->外2",
            "part": "glass",
            "area": 1.0,
            "u_value": 5.0,
            "SCR": 0.6,
            "SCC": 0.2,
            "solar": [80.0, 0.0],
        },
    ]
    receivers = [
        {"key": "A->床", "part": "floor", "area": 12.0, "u_value": 1.0, "eta": 0.7},
        {"key": "A->壁", "part": "wall", "area": 18.0, "u_value": 1.0, "eta": 0.6},
        {"key": "A->天井", "part": "ceiling", "area": 12.0, "u_value": 1.0, "eta": 0.5},
    ]
    surfaces = windows + receivers
    all_gains = []
    for w in windows:
        all_gains.extend(process_glass_solar(w, surfaces, sim_length=2, verbose=False))

    for t in (0, 1):
        expected_rad = sum(w["solar"][t] * w["area"] * w["SCR"] for w in windows)
        expected_conv = sum(w["solar"][t] * w["area"] * w["SCC"] for w in windows)
        surface = sum(b["heat_generation"][t] for b in all_gains if b["key"] != "void->A")
        air = sum(b["heat_generation"][t] for b in all_gains if b["key"] == "void->A")
        assert surface == pytest.approx(expected_rad)
        assert air == pytest.approx(expected_conv)


def test_whole_window_u_still_conserved():
    window = {
        "key": "A->B",
        "part": "glass",
        "area": 2,
        "whole_window_u": 4.65,
        "SCC": 0.158,
        "SCR": 0.474,
        "solar": [500, 0],
    }
    surfaces = [
        window,
        {"key": "A->C", "part": "wall", "area": 4, "u_value": 1, "eta": 0.65},
        {"key": "A->D", "part": "floor", "area": 6, "u_value": 1},
    ]
    _, branches = process_surfaces(surfaces, sim_length=2)
    gains = _gains(branches)
    assert sum(b["heat_generation"][0] for b in gains if b["key"] != "void->A") == pytest.approx(474)
    assert sum(b["heat_generation"][0] for b in gains if b["key"] == "void->A") == pytest.approx(158)


def test_non_conserving_distribution_rejected():
    window = {
        "key": "A->外",
        "part": "glass",
        "area": 1.0,
        "u_value": 5.0,
        "SCR": 0.5,
        "SCC": 0.0,
        "solar": [10.0],
        "scr_distribution": "legacy",
    }
    with pytest.raises(ValueError, match="scr_distribution"):
        process_glass_solar(window, [window], sim_length=1, verbose=False)

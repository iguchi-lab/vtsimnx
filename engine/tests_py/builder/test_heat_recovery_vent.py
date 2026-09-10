import pytest

from app.builder import build_config


def _base_sim():
    return {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T01:00:00Z",
                "timestep": 3600,
                "length": 2,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": True, "x": True, "c": False},
        },
        "nodes": [
            {"key": "外部", "t": 0.0, "x": 0.003},
            {"key": "室1", "calc_t": True, "calc_x": True, "t": 20.0, "x": 0.008, "v": 50.0, "thermal_mass": 60000.0},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
    }


def test_hrv_sensible_vol_expands_nodes_and_branches():
    raw = _base_sim()
    raw["heat_recovery_vent"] = [
        {
            "key": "HRV1",
            "outdoor": "外部",
            "room": "室1",
            "recovery": "sensible",
            "eta_t": 0.75,
            "vol": 0.05,
        }
    ]
    out = build_config(raw, add_surface=False, add_capacity=False, add_moisture_capacity=False)
    keys = {n["key"] for n in out["nodes"]}
    assert "HRV1" in keys
    assert "HRV1_exhaust" in keys
    hrv = next(n for n in out["nodes"] if n["key"] == "HRV1")
    assert hrv["type"] == "hrv"
    assert hrv["calc_t"] is False
    assert hrv["calc_x"] is False
    assert hrv["ac_spec"]["eta_t"] == 0.75
    assert hrv["ac_spec"]["eta_x"] == 0.0
    assert hrv["ac_spec"]["recovery"] == "sensible"
    vkeys = {b["key"] for b in out["ventilation_branches"]}
    assert "外部->HRV1" in vkeys
    assert "HRV1->室1" in vkeys
    assert "室1->HRV1_exhaust" in vkeys
    assert "HRV1_exhaust->外部" in vkeys
    for k in ("外部->HRV1", "HRV1->室1", "室1->HRV1_exhaust", "HRV1_exhaust->外部"):
        b = next(x for x in out["ventilation_branches"] if x["key"] == k)
        assert b["type"] == "fixed_flow"
        assert b["subtype"] == "hrv"
        assert b["vol"] == [0.05, 0.05] or b["vol"] == 0.05 or b["vol"][0] == 0.05


def test_hrv_total_fan_pq_expands_supply_and_exhaust_fans():
    raw = _base_sim()
    raw["heat_recovery_vent"] = [
        {
            "key": "ERV1",
            "outdoor": "外部",
            "room": "室1",
            "recovery": "total",
            "eta_t": 0.7,
            "eta_x": 0.55,
            "vol": 0.1,
            "supply": {"p_max": 100.0, "p1": 50.0, "q1": 0.05, "q_max": 0.12},
            "exhaust": {"p_max": 90.0, "p1": 45.0, "q1": 0.04, "q_max": 0.11},
            "area": 0.06,
            "k_total": 1.5,
        }
    ]
    out = build_config(raw, add_surface=False, add_capacity=False, add_moisture_capacity=False)
    sa = next(n for n in out["nodes"] if n["key"] == "ERV1")
    assert sa["calc_p"] is True
    assert sa["ac_spec"]["eta_x"] == 0.55
    assert sa["ac_spec"]["recovery"] == "total"
    intake = next(b for b in out["ventilation_branches"] if b["key"] == "外部->ERV1")
    supply = next(b for b in out["ventilation_branches"] if b["key"] == "ERV1->室1")
    exhaust = next(b for b in out["ventilation_branches"] if b["key"] == "室1->ERV1_exhaust")
    outlet = next(b for b in out["ventilation_branches"] if b["key"] == "ERV1_exhaust->外部")
    assert intake["type"] == "fan" and intake["p_max"] == 100.0
    assert exhaust["type"] == "fan" and exhaust["p_max"] == 90.0
    assert supply["type"] == "pressure_loss" and supply["k_total"] == 1.5
    assert outlet["type"] == "pressure_loss"
    assert "vol" not in intake
    assert out["simulation"]["calc_flag"]["p"] is True


def test_hrv_separate_oa_sa_ra_ea_ports():
    raw = _base_sim()
    raw["nodes"].extend(
        [
            {"key": "給気チャンバ", "calc_t": True, "t": 18.0, "v": 5.0, "thermal_mass": 6000.0},
            {"key": "還気チャンバ", "calc_t": True, "t": 21.0, "v": 5.0, "thermal_mass": 6000.0},
            {"key": "排気口", "t": 5.0},
        ]
    )
    raw["heat_recovery_vent"] = [
        {
            "key": "HRV1",
            "oa": "外部",
            "sa": "給気チャンバ",
            "ra": "還気チャンバ",
            "ea": "排気口",
            "recovery": "total",
            "eta_t": 0.8,
            "eta_x": 0.5,
            "vol": 0.04,
        }
    ]
    out = build_config(raw, add_surface=False, add_capacity=False, add_moisture_capacity=False)
    hrv = next(n for n in out["nodes"] if n["key"] == "HRV1")
    assert hrv["outside_node"] == "外部"
    assert hrv["in_node"] == "還気チャンバ"
    assert hrv["set_node"] == "給気チャンバ"
    assert hrv["ac_spec"]["oa_node"] == "外部"
    assert hrv["ac_spec"]["sa_node"] == "給気チャンバ"
    assert hrv["ac_spec"]["ra_node"] == "還気チャンバ"
    assert hrv["ac_spec"]["ea_node"] == "排気口"
    vkeys = {b["key"] for b in out["ventilation_branches"]}
    assert "外部->HRV1" in vkeys
    assert "HRV1->給気チャンバ" in vkeys
    assert "還気チャンバ->HRV1_exhaust" in vkeys
    assert "HRV1_exhaust->排気口" in vkeys


def test_hrv_rejects_partial_pq():
    raw = _base_sim()
    raw["heat_recovery_vent"] = [
        {
            "key": "HRV1",
            "outdoor": "外部",
            "room": "室1",
            "recovery": "sensible",
            "supply": {"p_max": 100.0, "p1": 50.0, "q1": 0.05, "q_max": 0.12},
        }
    ]
    with pytest.raises(ValueError, match="supply/exhaust"):
        build_config(raw, add_surface=False, add_capacity=False, add_moisture_capacity=False)

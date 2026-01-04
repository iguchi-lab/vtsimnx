from __future__ import annotations

from app.builder import build_config_with_warning_details


def test_heat_source_adds_heat_generation_branch_to_room():
    raw = {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T00:01:00Z",
                "timestep": 60,
                "length": 2,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [{"key": "LD", "t": 20.0, "calc_t": True}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "heat_source": [
            {"key": "LD_人体", "room": "LD", "generation_rate": [100.0, 200.0], "convection": 1.0, "radiation": 0.0}
        ],
    }

    built, warnings, warning_details = build_config_with_warning_details(raw, output_path=None)
    assert isinstance(warnings, list)
    assert isinstance(warning_details, list)

    tb = built.get("thermal_branches") or []
    assert any(b.get("key") == "void->LD" and b.get("type") == "heat_generation" for b in tb)


def test_heat_source_radiation_distributes_to_surfaces_when_available():
    raw = {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T00:01:00Z",
                "timestep": 60,
                "length": 1,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        },
        "nodes": [{"key": "LD", "t": 20.0, "calc_t": True}, {"key": "外部", "t": 0.0, "calc_t": False}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {"key": "LD->外部", "part": "wall", "area": 10.0, "u_value": 1.0},
            {"key": "LD->外部", "part": "floor", "area": 5.0, "u_value": 1.0},
        ],
        "heat_source": [
            {"key": "LD_人体", "room": "LD", "generation_rate": 300.0, "convection": 0.0, "radiation": 1.0}
        ],
    }

    built, _warnings, _details = build_config_with_warning_details(raw, output_path=None)
    tb = built.get("thermal_branches") or []

    # 表面ノードへの heat_generation が作られている（2面なので2本）
    hs = [b for b in tb if b.get("type") == "heat_generation" and str(b.get("subtype")) == "internal_radiation"]
    assert len(hs) == 2
    assert all(str(b.get("key", "")).startswith("void->") for b in hs)



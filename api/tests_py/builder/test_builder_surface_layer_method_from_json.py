from app.builder import build_config


def test_builder_surface_layer_method_from_json_applies_to_all_surfaces():
    raw = {
        "builder": {"surface_layer_method": "response"},
        "simulation": {
            "index": {
                "start": "2000-01-01T00:00:00",
                "end": "2000-01-01T00:00:00",
                "timestep": 60,
                "length": 2,
            },
            "tolerance": {"ventilation": 1e-3, "thermal": 1e-3, "convergence": 1e-6},
        },
        "nodes": [{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "A->B",
                "part": "wall",
                "area": 10.0,
                "layers": [{"lambda": 0.8, "t": 0.12, "v_capa": 900000.0}],
                # layer_method を書かない（JSON一括指定で response になる想定）
            }
        ],
        "aircon": [],
    }

    out = build_config(raw, output_path=None)
    tb = out.get("thermal_branches", [])
    assert any(isinstance(b, dict) and b.get("type") == "response_conduction" for b in tb)



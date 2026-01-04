import json

from app.builder import build_config


def test_build_config_minimal(tmp_path, minimal_input_config):
    out_path = tmp_path / "parsed.json"
    out = build_config(minimal_input_config, output_path=str(out_path))
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    # 基本キーが揃っている
    assert set(saved.keys()) >= {"simulation", "nodes", "ventilation_branches", "thermal_branches"}
    # 返り値も辞書で一致（キーの一部だけ検証）
    assert "simulation" in out and "nodes" in out


def test_builder_surface_layer_method_argument_overrides_json_builder_option():
    # JSON側で response を指定していても、引数で明示したら引数が優先される（関数引数がrc以外のときはJSONを読まない仕様）
    raw = {
        "builder": {"surface_layer_method": "response"},
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 2},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "A->B",
                "part": "wall",
                "area": 10.0,
                "u_value": 0.5,
            }
        ],
        "aircon": [],
    }

    out = build_config(raw, output_path=None, surface_layer_method="rc")
    # RCなら conductance ブランチができる。responseなら response_conduction ができる。
    assert any(b.get("subtype") == "conduction" and b.get("conductance") is not None for b in out.get("thermal_branches", []))
    assert not any(b.get("type") == "response_conduction" for b in out.get("thermal_branches", []))



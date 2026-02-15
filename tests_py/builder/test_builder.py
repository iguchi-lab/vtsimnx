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


def test_builder_flags_can_be_set_in_json_builder_section_and_overridden_by_args():
    raw = {
        "builder": {
            # builder内で surfaces 展開のON/OFFを制御できる
            "add_surface": False,
        },
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室内", "t": 22.0}, {"key": "外部", "t": 5.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [{"key": "室内->外部", "part": "wall", "area": 2.0, "u_value": 1.0}],
    }

    # builder.add_surface=false なので surface由来の layer ノードが追加されない
    out1 = build_config(raw, output_path=None, add_aircon=False, add_capacity=False, add_surface=None)
    node_keys1 = {n["key"] for n in out1["nodes"]}
    assert "室内-外部_wall_s" not in node_keys1

    # 引数で明示すれば上書きできる
    out2 = build_config(raw, output_path=None, add_aircon=False, add_capacity=False, add_surface=True)
    node_keys2 = {n["key"] for n in out2["nodes"]}
    assert "室内-外部_wall_s" in node_keys2


def test_builder_can_disable_nocturnal_loss_generation_and_be_overridden_by_args():
    raw = {
        "builder": {
            "add_surface": True,
            "add_surface_nocturnal": False,
        },
        "simulation": {
            "index": {"start": "2000-01-01T00:00:00", "end": "2000-01-01T00:00:00", "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [{"key": "室内", "t": 22.0}, {"key": "外部", "t": 5.0}],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            {
                "key": "室内->外部",
                "part": "wall",
                "area": 2.0,
                "u_value": 1.0,
                # 夜間放射（W/m2）を入れておく
                "nocturnal": [10.0],
            }
        ],
    }

    out1 = build_config(raw, output_path=None, add_aircon=False, add_capacity=False, add_surface=None)
    assert not any(b.get("subtype") == "nocturnal_loss" for b in out1.get("thermal_branches", []))

    out2 = build_config(
        raw,
        output_path=None,
        add_aircon=False,
        add_capacity=False,
        add_surface=True,
        add_surface_nocturnal=True,
    )
    assert any(b.get("subtype") == "nocturnal_loss" for b in out2.get("thermal_branches", []))



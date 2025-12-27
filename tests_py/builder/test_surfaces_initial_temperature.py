from app.builder import build_config


def test_surface_generated_layer_nodes_copy_initial_temperature_from_leading_node_key():
    raw = {
        "simulation": {
            "index": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T01:00:00Z",
                "timestep": 60,
                "length": 1,
            },
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
            "calc_flag": {"p": False, "t": False, "x": False, "c": False},
        },
        "nodes": [
            {"key": "室内", "t": 22.0},
            {"key": "外部", "t": 5.0},
        ],
        "ventilation_branches": [],
        "thermal_branches": [],
        "surfaces": [
            # 単層（u_value）: 内側/外側の表面ノードが2つできる
            {"key": "室内->外部", "part": "wall", "area": 2.0, "u_value": 1.0},
            # 多層（layers）: 内部ノードも追加される（先頭ノードは "室内"）
            {
                # 同じ key だとノード/ブランチが重複してバリデーションで落ちるためコメントで区別
                "key": "室内->外部||multi",
                "part": "wall",
                "area": 2.0,
                "layers": [
                    {"t": 0.1, "lambda": 1.0, "v_capa": 1.0},
                    {"t": 0.1, "lambda": 1.0, "v_capa": 1.0},
                ],
            },
        ],
    }

    out = build_config(raw, add_aircon=False, add_capacity=False, add_surface=True)
    nodes = {n["key"]: n for n in out["nodes"]}

    # 単層の表面ノード
    assert nodes["室内-外部_wall_s"]["t"] == 22.0
    assert nodes["外部-室内_wall_s"]["t"] == 5.0

    # 多層の内部ノード（キーの先頭が "室内" なので 22.0 をコピー）
    assert nodes["室内-外部(multi)_wall_1-2"]["t"] == 22.0



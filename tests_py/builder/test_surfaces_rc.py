from app.builder.surfaces import process_surfaces


def test_surfaces_rc_layers_generates_layer_nodes_and_branch_chain():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [
                {"lambda": 1.0, "t": 0.10, "v_capa": 1000.0},
                {"lambda": 0.5, "t": 0.20, "v_capa": 2000.0},
            ],
            # layer_method 省略（デフォルト rc）
        }
    ]

    nodes, tbs = process_surfaces(
        surfaces,
        sim_length=2,
        node_config=[{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
        add_solar=False,
        add_radiation=False,
        time_step=60.0,
    )

    # RC: 層数n=2 → layerノードは n+1=3（surface, internal, surface）
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 3

    # thermal_branches: (対流 + 伝導*n + 対流) = n+2 = 4
    assert len(tbs) == 4
    assert [b.get("subtype") for b in tbs] == ["convection", "conduction", "conduction", "convection"]

    # 伝導の conductance は area*lambda/t
    # 1層目: 10*1/0.1=100
    assert abs(tbs[1]["conductance"] - 100.0) < 1e-9
    # 2層目: 10*0.5/0.2=25
    assert abs(tbs[2]["conductance"] - 25.0) < 1e-9


def test_surfaces_rc_u_value_generates_single_conduction_and_optional_capacity():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "u_value": 0.5,
            "a_capacity": 20.0,  # areaあたりではなく、この実装では areaを掛けて thermal_mass に変換される
        }
    ]

    nodes, tbs = process_surfaces(
        surfaces,
        sim_length=2,
        node_config=[{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
        add_solar=False,
        add_radiation=False,
        time_step=60.0,
    )

    # n=1 扱い → layerノードは 2（surface, surface）
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 2

    # branches は 3（対流, 伝導, 対流）
    assert len(tbs) == 3
    assert [b.get("subtype") for b in tbs] == ["convection", "conduction", "convection"]

    # 伝導は conductance=area*u_value
    assert abs(tbs[1]["conductance"] - (10.0 * 0.5)) < 1e-9



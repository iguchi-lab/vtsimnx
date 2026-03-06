import pytest

from app.builder.surfaces import process_surfaces
from app.builder.validate import ConfigFileError


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


def test_surfaces_invalid_part_raises_config_error():
    surfaces = [
        {
            "key": "A->B",
            "part": "roof",
            "area": 10.0,
            "u_value": 0.5,
        }
    ]

    with pytest.raises(ConfigFileError):
        process_surfaces(
            surfaces,
            sim_length=2,
            add_solar=False,
            add_radiation=False,
            time_step=60.0,
        )


def test_surfaces_part_accepts_case_and_whitespace():
    surfaces = [
        {
            "key": "A->B",
            "part": " Wall ",
            "area": 10.0,
            "u_value": 0.5,
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

    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 2
    assert [b.get("subtype") for b in tbs] == ["convection", "conduction", "convection"]


def test_surfaces_part_accepts_window_alias_as_glass():
    surfaces = [
        {
            "key": "A->B",
            "part": "window",
            "area": 10.0,
            "u_value": 0.5,
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

    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 2
    assert [b.get("subtype") for b in tbs] == ["convection", "conduction", "convection"]


def test_surfaces_rc_hollow_layer_can_use_thermal_resistance():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [
                {"air_layer": True, "thermal_resistance": 0.20},
            ],
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

    # n=1扱いのため表面ノードは2つ
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 2

    # 対流・中空層相当の伝導・対流
    assert [b.get("subtype") for b in tbs] == ["convection", "conduction", "convection"]
    # conductance = area / R
    assert abs(tbs[1]["conductance"] - (10.0 / 0.20)) < 1e-9


def test_surfaces_rc_hollow_layer_with_thickness_adds_air_capacity_node():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [
                {"air_layer": True, "thermal_resistance": 0.20, "t": 0.05},
            ],
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

    # 両端2ノード + 中心ノード1つ（空気熱容量）
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 3
    center = [n for n in layer_nodes if "_air" in n.get("key", "")]
    assert len(center) == 1
    assert abs(center[0]["thermal_mass"] - (10.0 * 0.05 * 1200.0)) < 1e-9

    # [室内側対流, 半抵抗伝導, 半抵抗伝導, 室外側対流]
    assert [b.get("subtype") for b in tbs] == [
        "convection",
        "conduction",
        "conduction",
        "convection",
    ]
    # 各半抵抗の conductance = 2 * area / R
    assert abs(tbs[1]["conductance"] - (2.0 * 10.0 / 0.20)) < 1e-9
    assert abs(tbs[2]["conductance"] - (2.0 * 10.0 / 0.20)) < 1e-9


def test_surfaces_rc_ventilated_layer_generates_center_node_and_three_internal_branches():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [
                {
                    "ventilated_air_layer": True,
                    "t": 0.05,
                    "alpha_c1": 3.0,
                    "alpha_c2": 4.0,
                    "alpha_r": 5.0,
                }
            ],
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

    # 両端2ノード + 中心ノード1つ
    layer_nodes = [n for n in nodes if n.get("type") == "layer"]
    assert len(layer_nodes) == 3
    center = [n for n in layer_nodes if "_vent" in n.get("key", "")]
    assert len(center) == 1
    # thermal_mass = area * t * air_v_capa(default=1200)
    assert abs(center[0]["thermal_mass"] - (10.0 * 0.05 * 1200.0)) < 1e-9

    # [室内側対流, c1対流, c2対流, 放射, 室外側対流]
    assert [b.get("subtype") for b in tbs] == [
        "convection",
        "convection",
        "convection",
        "radiation",
        "convection",
    ]
    assert abs(tbs[1]["conductance"] - (10.0 * 3.0)) < 1e-9
    assert abs(tbs[2]["conductance"] - (10.0 * 4.0)) < 1e-9
    assert abs(tbs[3]["conductance"] - (10.0 * 5.0)) < 1e-9


def test_surfaces_rc_legacy_layer_flags_are_not_supported():
    surfaces = [
        {
            "key": "A->B",
            "part": "wall",
            "area": 10.0,
            "layers": [
                {"is_hollow": True, "thermal_resistance": 0.20},
            ],
        }
    ]

    with pytest.raises(ValueError):
        process_surfaces(
            surfaces,
            sim_length=2,
            node_config=[{"key": "A", "t": 20.0}, {"key": "B", "t": 0.0}],
            add_solar=False,
            add_radiation=False,
            time_step=60.0,
        )



"""materials 公開テーブルの不変化・単位・代表値の回帰テスト。"""
from __future__ import annotations

from types import MappingProxyType

import pytest

import vtsimnx as vt
from vtsimnx.materials import copy_materials, get_material, materials
from vtsimnx.materials.table import _materials_kj_per_m3k, materials as table_materials


def test_materials_is_mapping_proxy():
    assert isinstance(materials, MappingProxyType)
    assert isinstance(vt.materials, MappingProxyType)


def test_materials_top_level_is_immutable():
    with pytest.raises(TypeError):
        materials["__should_not_write__"] = {"lambda": 1.0, "v_capa": 1.0}  # type: ignore[index]


def test_materials_row_is_immutable():
    row = materials["合板"]
    assert isinstance(row, MappingProxyType)
    with pytest.raises(TypeError):
        row["lambda"] = 0.0  # type: ignore[index]


def test_materials_count_and_required_keys():
    assert len(materials) >= len(table_materials)
    assert len(materials) >= 91
    for name, props in materials.items():
        assert set(props.keys()) == {"lambda", "v_capa"}, name
        assert props["lambda"] > 0.0, name
        assert props["v_capa"] > 0.0, name


def test_table_materials_v_capa_is_j_per_m3k():
    """table 由来の v_capa は内部 kJ/(m³·K) を ×1000 した J/(m³·K)。"""
    for name, kj_props in _materials_kj_per_m3k.items():
        pub = materials[name]
        assert pub["lambda"] == pytest.approx(kj_props["lambda"])
        assert pub["v_capa"] == pytest.approx(kj_props["v_capa"] * 1000.0)


def test_representative_materials_regression():
    plywood = materials["合板"]
    assert plywood["lambda"] == pytest.approx(0.16)
    assert plywood["v_capa"] == pytest.approx(715806.0)

    concrete = materials["コンクリート"]
    assert concrete["lambda"] == pytest.approx(1.6)
    assert concrete["v_capa"] == pytest.approx(1896260.0)

    gypsum = materials["せっこうボード"]
    assert gypsum["lambda"] == pytest.approx(0.22)
    assert gypsum["v_capa"] == pytest.approx(904176.0)

    cavity = materials["中空層(1cm以上)"]
    assert cavity["lambda"] == pytest.approx(11.11)
    assert cavity["v_capa"] == pytest.approx(1298.0)


def test_get_material_returns_mutable_copy():
    props = get_material("合板")
    assert props["lambda"] == pytest.approx(0.16)
    assert props["v_capa"] == pytest.approx(715806.0)
    props["lambda"] = 9.9
    assert materials["合板"]["lambda"] == pytest.approx(0.16)


def test_get_material_unknown_raises():
    with pytest.raises(KeyError, match="unknown material"):
        get_material("__no_such_material__")


def test_copy_materials_is_mutable_and_isolated():
    local = copy_materials()
    local["自作断熱材"] = {"lambda": 0.03, "v_capa": 42000.0}
    assert "自作断熱材" not in materials
    local["合板"]["lambda"] = 0.0
    assert materials["合板"]["lambda"] == pytest.approx(0.16)


def test_spread_material_props_still_works():
    layer = {"key": "合板", **materials["合板"], "t": 0.012}
    assert layer["key"] == "合板"
    assert layer["lambda"] == pytest.approx(0.16)
    assert layer["v_capa"] == pytest.approx(715806.0)
    assert layer["t"] == 0.012

"""
surfaces モジュールの内部ヘルパー（_get_air_v_capa, _build_layer_node_dict,
_apply_*_layer, _layers_to_rc_arrays, _build_rc_continuous_abcd）のユニットテスト。
"""
import numpy as np
import pytest

from app.builder.surfaces import (
    DEFAULT_AIR_V_CAPA,
    _get_air_v_capa,
    _build_layer_node_dict,
    _apply_ventilated_layer,
    _apply_hollow_layer,
    _apply_normal_layer,
    _layers_to_rc_arrays,
    _build_rc_continuous_abcd,
)


# --- _get_air_v_capa ---


def test_get_air_v_capa_returns_default_when_empty():
    assert _get_air_v_capa({}) == DEFAULT_AIR_V_CAPA


def test_get_air_v_capa_returns_explicit_air_v_capa():
    assert _get_air_v_capa({"air_v_capa": 1500.0}) == 1500.0


def test_get_air_v_capa_accepts_v_capa_air_alias():
    assert _get_air_v_capa({"v_capa_air": 1400.0}) == 1400.0


def test_get_air_v_capa_returns_default_when_negative():
    assert _get_air_v_capa({"air_v_capa": -1.0}) == DEFAULT_AIR_V_CAPA


# --- _build_layer_node_dict ---


def test_build_layer_node_dict_has_required_keys():
    d = _build_layer_node_dict("A-B_s", "surface", 100.0, None)
    assert d["key"] == "A-B_s"
    assert d["subtype"] == "surface"
    assert d["thermal_mass"] == 100.0
    assert d["calc_t"] is True
    assert d["type"] == "layer"
    assert "t" not in d


def test_build_layer_node_dict_sets_initial_t_when_lead_in_config():
    d = _build_layer_node_dict("A-B_wall_s", "surface", 50.0, {"A": 20.0})
    assert d["t"] == 20.0


def test_build_layer_node_dict_no_t_when_lead_not_in_config():
    d = _build_layer_node_dict("A-B_wall_s", "surface", 50.0, {"B": 10.0})
    assert "t" not in d


# --- _apply_hollow_layer ---


def test_apply_hollow_layer_returns_half_capacity_and_conduction():
    layer = {"thermal_resistance": 0.2, "t": 0.05}
    add_left, add_right, branches = _apply_hollow_layer(
        layer, 0, "L", "R", 10.0, "pre", "S"
    )
    # 10 * 0.05 * 1298 / 2 = 324.5
    assert abs(add_left - 324.5) < 1e-9
    assert abs(add_right - 324.5) < 1e-9
    assert len(branches) == 1
    assert branches[0]["key"] == "L->R"
    assert branches[0]["subtype"] == "conduction"
    assert abs(branches[0]["conductance"] - (10.0 / 0.2)) < 1e-9


def test_apply_hollow_layer_raises_when_t_missing():
    with pytest.raises(ValueError, match="requires positive 't'"):
        _apply_hollow_layer(
            {"thermal_resistance": 0.2}, 0, "L", "R", 10.0, "pre", "S"
        )


def test_apply_hollow_layer_raises_when_resistance_missing():
    with pytest.raises(ValueError, match="requires.*thermal_resistance"):
        _apply_hollow_layer(
            {"t": 0.05}, 0, "L", "R", 10.0, "pre", "S"
        )


# --- _apply_ventilated_layer ---


def test_apply_ventilated_layer_returns_center_node_and_three_branches():
    layer = {"t": 0.05, "alpha_c1": 3.0, "alpha_c2": 4.0, "alpha_r": 5.0}
    extra, branches = _apply_ventilated_layer(
        layer, 0, "L", "R", 10.0, "pre", "S"
    )
    assert len(extra) == 1
    assert extra[0][0] == "pre_1_vent"
    assert extra[0][1] == "internal"
    assert abs(extra[0][2] - (10.0 * 0.05 * DEFAULT_AIR_V_CAPA)) < 1e-9
    assert len(branches) == 3
    assert branches[0]["subtype"] == "convection"
    assert branches[1]["subtype"] == "convection"
    assert branches[2]["subtype"] == "radiation"


def test_apply_ventilated_layer_raises_when_t_missing():
    with pytest.raises(ValueError, match="requires positive 't'"):
        _apply_ventilated_layer(
            {"alpha_c1": 3.0}, 0, "L", "R", 10.0, "pre", "S"
        )


# --- _apply_normal_layer ---


def test_apply_normal_layer_returns_half_capacity_and_conduction():
    layer = {"lambda": 1.0, "t": 0.1, "v_capa": 1000.0}
    add_left, add_right, branches = _apply_normal_layer(
        layer, 0, "L", "R", 10.0, "S"
    )
    # 10 * 1000 * 0.1 / 2 = 500
    assert abs(add_left - 500.0) < 1e-9
    assert abs(add_right - 500.0) < 1e-9
    assert len(branches) == 1
    assert branches[0]["conductance"] == 10.0 * 1.0 / 0.1


def test_apply_normal_layer_raises_when_lambda_missing():
    with pytest.raises(ValueError, match="requires lambda, t, v_capa"):
        _apply_normal_layer(
            {"t": 0.1, "v_capa": 1000.0}, 0, "L", "R", 10.0, "S"
        )


# --- _layers_to_rc_arrays ---


def test_layers_to_rc_arrays_returns_arrays():
    layers = [
        {"lambda": 1.0, "t": 0.1, "v_capa": 1000.0},
        {"lambda": 0.5, "t": 0.2, "v_capa": 2000.0},
    ]
    n, lam, thk, vc, C, R_half, R_between = _layers_to_rc_arrays(layers)
    assert n == 2
    assert len(lam) == 2
    assert len(R_between) == 1
    assert abs(C[0] - 100.0) < 1e-9
    assert abs(C[1] - 400.0) < 1e-9


def test_layers_to_rc_arrays_raises_when_empty():
    with pytest.raises(ValueError, match="layers is empty"):
        _layers_to_rc_arrays([])


def test_layers_to_rc_arrays_raises_when_invalid_lambda():
    with pytest.raises(ValueError, match="invalid layer properties"):
        _layers_to_rc_arrays([{"lambda": 0.0, "t": 0.1, "v_capa": 1000.0}])


# --- _build_rc_continuous_abcd ---


def test_build_rc_continuous_abcd_single_layer():
    n = 1
    C = np.array([100.0])
    R_half = np.array([0.05])
    R_between = np.array([])
    A, B, Cmat, Dmat = _build_rc_continuous_abcd(n, C, R_half, R_between)
    assert A.shape == (1, 1)
    assert B.shape == (1, 2)
    assert Cmat.shape == (2, 1)
    assert Dmat.shape == (2, 2)
    # 1層: g_s = g_t = 1/0.05 = 20, A[0,0] = -(20+20)/100 = -0.4
    assert abs(A[0, 0] - (-0.4)) < 1e-9


def test_build_rc_continuous_abcd_two_layers():
    n = 2
    C = np.array([100.0, 200.0])
    R_half = np.array([0.05, 0.1])
    R_between = np.array([0.15])
    A, B, Cmat, Dmat = _build_rc_continuous_abcd(n, C, R_half, R_between)
    assert A.shape == (2, 2)
    assert B.shape == (2, 2)
    assert Cmat.shape == (2, 2)
    assert Dmat.shape == (2, 2)

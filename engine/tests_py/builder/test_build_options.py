"""BuildOptions 解決の回帰テスト。"""
from __future__ import annotations

import pytest

from app.builder.build_options import BuildOptions


def test_build_options_defaults():
    opt = BuildOptions.resolve({})
    assert opt.add_surface is True
    assert opt.add_aircon is True
    assert opt.add_capacity is True
    assert opt.add_moisture_capacity is True
    assert opt.add_surface_solar is True
    assert opt.add_surface_nocturnal is True
    assert opt.add_surface_radiation is True
    assert opt.add_surface_radiation_exclude_glass is False
    assert opt.surface_layer_method == "rc"
    assert opt.response_method == "arx_rc"
    assert opt.response_terms is None


def test_build_options_prefers_function_args():
    opt = BuildOptions.resolve(
        {"builder": {"add_surface": True}, "add_aircon": True},
        add_surface=False,
        add_aircon=False,
    )
    assert opt.add_surface is False
    assert opt.add_aircon is False


def test_build_options_reads_builder_then_toplevel():
    opt = BuildOptions.resolve(
        {
            "builder": {"add_surface": False, "add_surface_radiation_exclude_glass": True},
            "add_aircon": False,
        }
    )
    assert opt.add_surface is False
    assert opt.add_aircon is False
    assert opt.add_surface_radiation_exclude_glass is True


def test_build_options_reads_layer_method_from_builder_when_default():
    opt = BuildOptions.resolve(
        {"builder": {"surface_layer_method": "response", "response_method": "ctf", "response_terms": 8}}
    )
    assert opt.surface_layer_method == "response"
    assert opt.response_method == "ctf"
    assert opt.response_terms == 8


def test_build_options_skips_json_layer_method_when_explicit():
    opt = BuildOptions.resolve(
        {"builder": {"surface_layer_method": "response", "response_method": "ctf", "response_terms": 8}},
        surface_layer_method="u_value",
    )
    assert opt.surface_layer_method == "u_value"
    # 従来どおり surface_layer_method 引数が既定以外なら JSON の response_* も読まない
    assert opt.response_method == "arx_rc"
    assert opt.response_terms is None


def test_build_options_invalid_response_terms():
    with pytest.raises(ValueError, match="builder.response_terms"):
        BuildOptions.resolve({"builder": {"response_terms": "x"}})

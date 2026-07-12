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


def test_build_options_reads_layer_method_from_builder_when_unspecified():
    opt = BuildOptions.resolve(
        {"builder": {"surface_layer_method": "response", "response_method": "ctf", "response_terms": 8}}
    )
    assert opt.surface_layer_method == "response"
    assert opt.response_method == "ctf"
    assert opt.response_terms == 8


def test_build_options_explicit_rc_overrides_json_response():
    # 明示的に "rc" を渡した場合は JSON の "response" に上書きされない
    opt = BuildOptions.resolve(
        {"builder": {"surface_layer_method": "response", "response_method": "ctf", "response_terms": 8}},
        surface_layer_method="rc",
    )
    assert opt.surface_layer_method == "rc"
    # response_* は独立解決: 未指定なら JSON を読む
    assert opt.response_method == "ctf"
    assert opt.response_terms == 8


def test_build_options_explicit_response_method_overrides_json():
    opt = BuildOptions.resolve(
        {"builder": {"response_method": "ctf", "response_terms": 8}},
        response_method="arx_rc",
        response_terms=3,
    )
    assert opt.response_method == "arx_rc"
    assert opt.response_terms == 3


def test_build_options_independent_string_resolution():
    # surface_layer_method だけ明示しても response_* の JSON 解決は阻害しない
    opt = BuildOptions.resolve(
        {"builder": {"surface_layer_method": "response", "response_method": "ctf", "response_terms": 8}},
        surface_layer_method="u_value",
    )
    assert opt.surface_layer_method == "u_value"
    assert opt.response_method == "ctf"
    assert opt.response_terms == 8


def test_build_options_invalid_response_terms():
    with pytest.raises(ValueError, match="builder.response_terms"):
        BuildOptions.resolve({"builder": {"response_terms": "x"}})

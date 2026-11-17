"""標準住宅の部品カタログ。気象は組み込まず、名前で選んで組み立てる。"""

from __future__ import annotations

import pandas as pd
import pytest

import vtsimnx as vt


def test_top_level_export():
    assert vt.standard_house is not None
    assert "standard_house" in vt.__all_experimental__
    names = vt.standard_house.available()
    assert names["ventilation"] == ["24h"]
    assert names["aircon"] == ["ac4"]
    assert "rac_standard" in names["aircon_spec"]


def test_rooms_and_layers_are_copies():
    h = vt.standard_house
    rooms = h.rooms["default"]
    assert rooms["LD"] == pytest.approx(51.67)
    rooms["LD"] = 0.0
    assert h.rooms["default"]["LD"] == pytest.approx(51.67)

    layers = h.layers["default"]
    assert "外壁_一般部" in layers
    assert layers["外壁_一般部"][0]["key"] == "せっこうボード"
    layers["外壁_一般部"][0]["t"] = 0.0
    assert h.layers["default"]["外壁_一般部"][0]["t"] == pytest.approx(0.0095)


def test_unknown_part_lists_available():
    with pytest.raises(KeyError, match="24h"):
        vt.standard_house.ventilation["gap"]


def test_ventilation_and_aircon_are_replaceable():
    h = vt.standard_house
    vent = h.ventilation["24h"]
    keys = [row["key"] for row in vent]
    assert "外部->LD->ホール" in keys
    vent.clear()
    assert len(h.ventilation["24h"]) > 0

    acs = h.aircon["ac4"]
    assert len(acs) == 4
    assert {unit["set"] for unit in acs} == {"LD", "寝室", "子供室1", "子供室2"}
    acs[:] = [acs[0]]
    assert len(h.aircon["ac4"]) == 4

    spec = h.aircon_spec["highspec"]
    spec["Q"]["heating"]["rtd"] = 99.0
    assert h.aircon_spec["highspec"]["Q"]["heating"]["rtd"] == pytest.approx(2.5)


def test_bind_surfaces_uses_caller_solar():
    h = vt.standard_house
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    dummy = pd.Series([0.0, 1.0, 2.0], index=idx)
    solar = {}
    nocturnal = {}
    types = h.surface_types["default"]
    for spec in types.values():
        if "solar_key" in spec:
            solar[spec["solar_key"]] = dummy
        if "nocturnal_key" in spec:
            nocturnal[spec["nocturnal_key"]] = dummy

    layers = h.layers["default"]
    catalog = h.bind_surface_types(types, layers, solar=solar, nocturnal=nocturnal)
    surfaces = h.compose_surfaces(h.surface_rows["default"], catalog)
    assert any(row["key"] == "LD->外部||S_窓_レース" for row in surfaces)
    ld_window = next(row for row in surfaces if row["key"] == "LD->外部||S_窓_レース")
    assert ld_window["area"] == pytest.approx(3.465 + 3.465)
    assert list(ld_window["solar"]) == [0.0, 1.0, 2.0]


def test_room_nodes_omit_weather_boundaries():
    nodes = vt.standard_house.room_nodes()
    keys = [node["key"] for node in nodes]
    assert "外部" not in keys
    assert "地下1m" not in keys
    assert keys[0] == "床下"
    assert keys[2] == "LD"
    ld = nodes[2]
    assert ld["calc_t"] is True
    assert ld["v"] == pytest.approx(51.67)

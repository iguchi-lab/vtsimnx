import pandas as pd
import pytest

import vtsimnx as vt


@pytest.mark.parametrize(
    "grade,r",
    [
        (4, 0.076 / 0.045),
        (5, 0.105 / 0.038 + 0.025 / 0.028),
        (6, 0.105 / 0.038 + 0.045 / 0.028),
        (7, 0.105 / 0.038 + 0.080 / 0.028),
    ],
)
def test_foundation(grade, r):
    h = vt.standard_house
    key = f"foundation{grade}"
    layers = h.layers[key]
    concrete, xps = layers["基礎外壁"]
    assert concrete["t"] == 0.120
    assert xps["t"] / xps["lambda"] == pytest.approx(r)
    assert layers["ホール-床下"] == [concrete]
    for row in h.surface_rows[key]:
        if "->床下" in row["key"] and row["surface"] != "ホール-床下":
            stack = layers[h.surface_types[key][row["surface"]]["layer_key"]]
            assert "合板" in stack[0]["key"]
            assert all("合板" in layer["key"] or "木材" in layer["key"] for layer in stack)


def test_gap_sweep():
    h = vt.standard_house
    branches = h.gap_network(1)
    fixed = h.gap_network(0)
    assert len([b for b in fixed if b.get("type") == "gap"]) == 156
    assert len(set(b["key"] for b in branches)) == len(branches)
    assert not any("床下" in b["key"] for b in branches)
    baseline = {b["key"]: b for b in fixed}
    for multiplier in h.GAP_CASES:
        trial = h.gap_network(multiplier)
        assert [b for b in trial if b["key"] in baseline] == fixed
        external = [b for b in trial if b["key"] not in baseline]
        assert len(external) == (228 if multiplier else 0)
        assert sum(b["a"] for b in external) == pytest.approx(
            multiplier * sum(b["a"] for b in branches if b["key"] not in baseline)
        )
    assert len(h.gap_network(central_air=True)) == len(branches) + 7
    assert h.central_aircon()[0]["in"] == "2階ホール"


def test_wind_and_rooms():
    h = vt.standard_house
    table = pd.DataFrame({f + "面": [0.5] * 16 for f in "東南西北"}, index=range(1, 17))
    weather = pd.DataFrame({"風向": [0, 3], "風速": [2.0, 2.0], "外気温": [10.0, 20.0]})
    p = h.facade_pressures(weather, table)
    assert p["南"] == pytest.approx([0, 101324.6 / (287.05 * 293.15)])
    nodes = h.gap_nodes([10.0, 20.0], [0.007] * 2, p, central_air=True)
    assert len([n for n in nodes if n.get("calc_p")]) == 14
    assert next(n for n in nodes if n["key"] == "東")["p"] == p["東"]

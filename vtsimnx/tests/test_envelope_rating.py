import pytest
import vtsimnx as vt


def test_south_window_wall_and_foundation():
    # 地域6。面積10+2+8、熱損失 5+4+3、南面のみが日射取得。
    data = {"surfaces": [{"key": key, "area": a} for key, a in [("壁", 10), ("窓", 2), ("土間", 8)]]}
    base = {"H": 1, "directions": ["S"]}
    assessment = {
        "region": 6,
        "parts": [
            dict(base, kind="wall", keys=["壁"], U=0.5),
            dict(base, kind="window", keys=["窓"], U=2, eta_d=0.6),
            {"kind": "slab", "keys": ["土間"]},
        ],
        "excluded": [],
        "lines": [{"name": "外周", "kind": "perimeter", "length": 5, "psi": 0.6, "H": 1}],
    }
    result = vt.envelope_performance(data, assessment)
    assert result["A_env"] == 20 and result["q"] == 12
    assert result["m_C"] == pytest.approx((0.17 + 2 * 0.6 * 0.93) * 0.434)
    assert result["m_H"] == pytest.approx((0.17 + 2 * 0.6 * 0.51) * 0.936)
    assert (result["UA"], result["eta_AC"], result["eta_AH"]) == (0.60, 2.8, 3.6)
    assessment["parts"][1]["eta_d_C"] = 0.4
    seasonal = vt.envelope_performance(data, assessment)
    assert seasonal["eta_AC"] < result["eta_AC"] and seasonal["eta_AH"] == result["eta_AH"]
    assessment["region"] = 8
    assert vt.envelope_performance(data, assessment)["eta_AH"] is None
    with pytest.raises(ValueError, match="perimeter"):
        vt.envelope_performance(data, {**assessment, "lines": []})


@pytest.mark.parametrize("model", list(vt.standard_house.surface_rows))
def test_standard_house_from_surface_input(model):
    h = vt.standard_house
    specs = h.surface_types[model]
    layers = h.layers[model]
    surfaces = [{**row, "layers": layers.get(specs[row["surface"]].get("layer_key"))} for row in h.surface_rows[model]]
    data = {"surfaces": surfaces}
    assessment = h.rating_template(model)
    slab_keys = {key for part in assessment["parts"] if part["kind"] == "slab" for key in part["keys"]}
    assert ("床下->地下1m" in slab_keys) == model.startswith("foundation")
    with pytest.raises(ValueError):
        vt.envelope_performance(data, assessment)
    for part in assessment["parts"]:
        if part["kind"] in {"window", "door"} and part["U"] is None:
            part["U"] = 3
        if part["kind"] == "window":
            part["eta_d"] = 0.5
    for line in assessment["lines"]:
        line.update(length=30, psi=0.5)
    result = vt.envelope_performance(data, assessment)
    assert result["UA"] > 0 and result["eta_AC"] > 0 and result["eta_AH"] > 0
    with pytest.raises(ValueError, match="mismatch"):
        vt.envelope_performance({"surfaces": [*surfaces, {"key": "追加面", "area": 1}]}, assessment)


def test_ventilated_wall_ends_at_air_layer():
    spec = {
        "region": 6,
        "parts": [{"kind": "wall", "keys": ["壁"], "outside": True, "H": 1, "directions": ["W"]}],
        "excluded": [],
        "lines": [],
    }
    data = {
        "surfaces": [
            {
                "key": "壁",
                "area": 10,
                "layers": [{"t": 0.1, "lambda": 0.05}, {"ventilated_air_layer": True}, {"t": 1, "lambda": 0.05}],
            }
        ]
    }
    result = vt.envelope_performance(data, spec)
    assert result["q"] == pytest.approx(10 / (0.11 + 2 + 0.11))


def test_window_specification_and_fallback():
    window = {"key": "南窓", "area": 2}
    part = {"kind": "window", "keys": ["南窓"], "U": 2, "H": 1, "directions": ["S"]}
    assessment = {"region": 6, "parts": [part], "lines": [], "excluded": []}

    def evaluate():
        return vt.envelope_performance({"surfaces": [window]}, assessment)["parts"][0]

    with pytest.warns(UserWarning, match="南窓.*frame_type"):
        result = evaluate()
    assert (result["eta_d_C"], result["eta_d_H"]) == (0.88, 0.0576)
    window.update(frame_type="金属製など", glass_type="一般複層", attachment="なし")
    assert evaluate()["eta_d_C"] == evaluate()["eta_d_H"] == 0.632
    part["attachment"] = "和障子"
    assert evaluate()["eta_d_C"] == 0.304
    del window["glass_type"]
    part["eta_d_C"] = 0.4
    with pytest.warns(UserWarning, match="glass_type"):
        result = evaluate()
    assert (result["eta_d_C"], result["eta_d_H"]) == (0.4, 0.176)
    assessment["region"] = 8
    assert "eta_d_H" not in evaluate()
    del part["eta_d_C"]
    part["glass_type"] = "入力誤り"
    with pytest.raises(ValueError, match="glass_type"):
        evaluate()


def test_perimeter_length_sources():
    import warnings

    part = {"kind": "slab", "keys": ["土間"]}
    wall = {"key": "立上り", "part": "wall", "area": 1.6}
    data = {"surfaces": [{"key": "土間", "area": 20}, wall]}
    line = {"name": "外周", "kind": "perimeter", "H": 0.7, "psi": 0.5, "wall_keys": ["立上り"]}
    assessment = {"region": 6, "parts": [part], "excluded": ["立上り"], "lines": [line]}

    def run():
        return vt.envelope_performance(data, assessment)["parts"][-1]

    with pytest.warns(UserWarning, match="外周.*0.4"):
        row = run()
    assert (row["length"], row["length_source"], row["q"]) == (4, "wall_keys", 1.4)
    line["wall_area"] = 2
    with pytest.warns(UserWarning):
        assert run()["length"] == 5
    line["length"] = 6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run()["length_source"] == "input"
    line["length"] = 0
    with pytest.raises(ValueError, match="length"):
        run()
    line.update(length=None, wall_area=None, wall_keys=[])
    with pytest.raises(ValueError, match="wall_area"):
        run()


def test_window_spec_to_rating_and_solver_inputs():
    spec = dict(frame_type="金属製", glass_type="一般複層", attachment="なし", gap_mm=6)
    props = vt.window_properties(**spec, convective_fraction=0.25)
    assert props["rating"] == {"U": 4.65, "eta_d": 0.632}
    assert props["surface"] == pytest.approx({"whole_window_u": 4.65, "SCC": 0.158, "SCR": 0.474})
    window = {"key": "窓", "area": 2}
    assessment = {"parts": [dict(spec, keys=["窓"], kind="window", H=1, directions=["S"])],
                  "region": 6, "excluded": [], "lines": []}
    assert vt.envelope_performance({"surfaces": [window]}, assessment)["q"] == pytest.approx(9.3)
    with pytest.warns(UserWarning, match="対流割合"), pytest.raises(ValueError, match="convective_fraction"):
        vt.window_properties(**spec)
    with pytest.raises(ValueError, match="適用範囲外"):
        vt.window_properties(**{**spec, "gap_mm": 3}, convective_fraction=0)
    props = vt.window_properties(**{**spec, "attachment": "和障子"}, SCC=0.1, SCR=0.2)
    assert props["rating"]["U"] == pytest.approx((4.65 + 1 / (1 / 4.65 + 0.18)) / 2)
    assert props["surface"]["SCC"] == 0.1 and props["surface"]["SCR"] == 0.2
    explicit = vt.window_properties(**{**spec, "attachment": "和障子"}, U=3.7, SCC=0.1, SCR=0.2)
    assert explicit["surface"]["whole_window_u"] == 3.7

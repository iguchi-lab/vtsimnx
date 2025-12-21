import pytest

pytest.importorskip("astropy")

import pandas as pd

import vtsimnx as vt


def test_make_solar_use_astro_runs():
    # 2点以上（make_solar内で delta を見るため）
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx, name="水平面全天日射量")

    df = vt.make_solar(全天日射量=s_ig, 緯度=36.0, 経度=140.0, use_astro=True)

    # 主要な出力列が作られること（値の厳密性まではここでは見ない）
    assert "日射熱取得量（南面）" in df.columns
    assert "太陽高度 hs" in df.columns
    assert "太陽方位角 AZs" in df.columns



import pandas as pd

import vtsimnx as vt


def test_solar_gain_by_angles_use_astro_runs():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)

    out = vt.solar_gain_by_angles(
        方位角=0.0,
        傾斜角=90.0,
        緯度=36.0,
        経度=140.0,
        全天日射量=s_ig,
        use_astro=True,
        名前="南面",
    )

    assert "太陽方位角 AZs" in out.columns
    assert "太陽高度 hs" in out.columns



import numpy as np
import pandas as pd

from vtsimnx.archenv.solar_position import sun_loc


def test_sun_loc_time_columns_match_legacy_definition():
    idx = pd.date_range("2026-01-01 00:30:00", periods=4, freq="6h")
    td = -0.5

    df = sun_loc(idx, lat=35.0, lon=135.0, td=td)

    # 旧実装定義:
    # N = (その年の1/1からの経過日数) + 1.5
    # H = 時 + 分/60 + td
    n_legacy = pd.Series([(t - pd.Timestamp(t.year, 1, 1)).days + 1.5 for t in idx], index=idx, dtype="float64")
    h_legacy = pd.Series([t.hour + t.minute / 60.0 + td for t in idx], index=idx, dtype="float64")

    assert np.allclose(df["元日からの通し日数 N"].to_numpy(dtype="float64"), n_legacy.to_numpy(dtype="float64"))
    assert np.allclose(df["時刻 H"].to_numpy(dtype="float64"), h_legacy.to_numpy(dtype="float64"))


def test_sun_loc_basic_output_columns_present():
    idx = pd.date_range("2026-06-01 00:00:00", periods=3, freq="1h")
    df = sun_loc(idx)

    required = {
        "元日からの通し日数 N",
        "時刻 H",
        "太陽高度 hs",
        "太陽方位角 AZs",
        "太陽高度の正弦 sin_hs",
        "太陽高度の余弦 cos_hs",
    }
    assert required.issubset(set(df.columns))
    assert df.index.equals(idx)

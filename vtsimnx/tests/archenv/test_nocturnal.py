import pandas as pd

import vtsimnx as vt


def test_make_nocturnal_accepts_dataframe_with_night_radiation():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    df_i = pd.DataFrame({"夜間放射量": [10.0, 20.0]}, index=idx)
    out = vt.make_nocturnal(df_i)
    assert "夜間放射量" in out.columns
    assert out["夜間放射量"].iloc[0] == 10.0


def test_make_nocturnal_accepts_dataframe_with_t_h():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    df_i = pd.DataFrame({"外気温": [10.0, 10.0], "外気相対湿度": [50.0, 50.0]}, index=idx)
    out = vt.make_nocturnal(df_i)
    assert "夜間放射量" in out.columns
    assert len(out) == 2



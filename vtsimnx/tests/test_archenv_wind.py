import pandas as pd
import pytest

import vtsimnx as vt


def test_make_wind_basic_output_shapes():
    idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1h")
    d = pd.Series([0, 4, 8], index=idx)  # 無風, E, S
    s = pd.Series([0.0, 2.0, 3.0], index=idx)

    df, wp = vt.make_wind(d, s)

    assert list(wp.keys()) == ["E", "S", "W", "N", "H"]
    assert df.index.equals(idx)
    assert wp["E"].index.equals(idx)
    assert wp["H"].index.equals(idx)
    assert (df["風圧_H"] <= 0.0).all()


def test_make_wind_rejects_negative_speed():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    d = pd.Series([1, 2], index=idx)
    s = pd.Series([1.0, -0.1], index=idx)

    with pytest.raises(ValueError, match="風速 s"):
        _ = vt.make_wind(d, s)


def test_make_wind_rejects_out_of_range_direction():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    d = pd.Series([0, 17], index=idx)
    s = pd.Series([1.0, 2.0], index=idx)

    with pytest.raises(ValueError, match="0..16"):
        _ = vt.make_wind(d, s)


def test_make_wind_requires_aligned_index():
    idx1 = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    idx2 = pd.date_range("2026-01-02 00:00:00", periods=2, freq="1h")
    d = pd.Series([1, 2], index=idx1)
    s = pd.Series([1.0, 2.0], index=idx2)

    with pytest.raises(ValueError, match="index"):
        _ = vt.make_wind(d, s)

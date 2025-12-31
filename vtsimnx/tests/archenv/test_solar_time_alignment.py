import pytest

import pandas as pd

import vtsimnx as vt


def test_make_solar_time_alignment_invalid():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx, name="水平面全天日射量")

    with pytest.raises(ValueError):
        _ = vt.solar_gain_by_angles(方位角=0.0, 傾斜角=90.0, 全天日射量=s_ig, time_alignment="bad")
    with pytest.raises(ValueError):
        _ = vt.solar_gain_by_angles(方位角=0.0, 傾斜角=90.0, 全天日射量=s_ig, timestamp_ref="bad")


def test_make_solar_time_alignment_modes_run():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx, name="水平面全天日射量")

    # index=区間開始想定（従来互換）
    _ = vt.solar_gain_by_angles(方位角=0.0, 傾斜角=90.0, 全天日射量=s_ig, time_alignment="center", timestamp_ref="start")
    # index=区間終了想定（質問のケース）
    _ = vt.solar_gain_by_angles(方位角=0.0, 傾斜角=90.0, 全天日射量=s_ig, time_alignment="center", timestamp_ref="end")
    # インデックスそのものを使う（td=0）
    _ = vt.solar_gain_by_angles(方位角=0.0, 傾斜角=90.0, 全天日射量=s_ig, time_alignment="timestamp", timestamp_ref="start")



import pandas as pd

import vtsimnx as vt


def test_nocturnal_gain_by_angles_accepts_horizontal_night_radiation():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    s_nh = pd.Series([10.0, 20.0], index=idx)

    out_v = vt.nocturnal_gain_by_angles(傾斜角=90.0, 夜間放射量_水平=s_nh, 名前="鉛直")
    out_h = vt.nocturnal_gain_by_angles(傾斜角=0.0, 夜間放射量_水平=s_nh, 名前="水平")

    assert "夜間放射量_水平" in out_v.columns
    assert "夜間放射量（鉛直）" in out_v.columns
    assert out_v["夜間放射量_水平"].iloc[0] == 10.0
    # 傾斜角=90° → 0.5倍（旧vertical_factor=0.5互換の一般化）
    assert out_v["夜間放射量（鉛直）"].iloc[0] == 5.0
    # 傾斜角=0° → 1.0倍
    assert out_h["夜間放射量（水平）"].iloc[0] == 10.0


def test_nocturnal_gain_by_angles_accepts_t_h():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    t = pd.Series([10.0, 10.0], index=idx)
    h = pd.Series([50.0, 50.0], index=idx)
    out = vt.nocturnal_gain_by_angles(傾斜角=90.0, 外気温=t, 外気相対湿度=h, 名前="鉛直")
    assert "夜間放射量_水平" in out.columns
    assert "夜間放射量（鉛直）" in out.columns
    assert len(out) == 2



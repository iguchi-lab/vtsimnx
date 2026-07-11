import pandas as pd

import vtsimnx as vt


def test_nocturnal_gain_by_angles_accepts_horizontal_night_radiation():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    s_nh = pd.Series([10.0, 20.0], index=idx)

    out_v = vt.nocturnal_gain_by_angles(tilt_deg=90.0, rn_horizontal=s_nh, return_details=True)
    out_h = vt.nocturnal_gain_by_angles(tilt_deg=0.0, rn_horizontal=s_nh, return_details=True)

    assert "nocturnal_radiation_horizontal" in out_v.columns
    assert "nocturnal_radiation" in out_v.columns
    assert out_v["nocturnal_radiation_horizontal"].iloc[0] == 10.0
    # 傾斜角=90° → 0.5倍（旧vertical_factor=0.5互換の一般化）
    assert out_v["nocturnal_radiation"].iloc[0] == 5.0
    # 傾斜角=0° → 1.0倍
    assert out_h["nocturnal_radiation"].iloc[0] == 10.0

    # 既定は Series を返す
    out_default = vt.nocturnal_gain_by_angles(tilt_deg=90.0, rn_horizontal=s_nh)
    assert isinstance(out_default, pd.Series)
    assert out_default.name == "nocturnal_radiation"


def test_nocturnal_gain_by_angles_accepts_t_h():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    t = pd.Series([10.0, 10.0], index=idx)
    h = pd.Series([50.0, 50.0], index=idx)
    out = vt.nocturnal_gain_by_angles(tilt_deg=90.0, t_out=t, rh_out=h, return_details=True)
    assert "nocturnal_radiation_horizontal" in out.columns
    assert "nocturnal_radiation" in out.columns
    assert len(out) == 2



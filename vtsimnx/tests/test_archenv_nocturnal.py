import pandas as pd
import pytest

import vtsimnx as vt
from vtsimnx.archenv.nocturnal import rn


def test_rn_returns_finite_values():
    v = rn(20.0, 50.0)
    assert isinstance(v, float)
    assert v == pytest.approx(v)  # not nan


def test_nocturnal_gain_by_angles_from_t_and_rh():
    idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1h")
    t_out = pd.Series([5.0, 6.0, 7.0], index=idx)
    rh_out = pd.Series([60.0, 65.0, 70.0], index=idx)

    s = vt.nocturnal_gain_by_angles(tilt_deg=90.0, t_out=t_out, rh_out=rh_out)

    assert isinstance(s, pd.Series)
    assert s.index.equals(idx)
    assert s.name == "夜間放射量"


def test_nocturnal_gain_by_angles_accepts_rn_horizontal():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    rn_h = pd.Series([40.0, 50.0], index=idx)

    out = vt.nocturnal_gain_by_angles(tilt_deg=0.0, rn_horizontal=rn_h, return_details=True)

    assert list(out.columns) == ["夜間放射量_水平", "夜間放射量"]
    assert out["夜間放射量"].tolist() == rn_h.tolist()


def test_nocturnal_gain_by_angles_rejects_mismatched_index():
    idx1 = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    idx2 = pd.date_range("2026-01-02 00:00:00", periods=2, freq="1h")
    t_out = pd.Series([5.0, 6.0], index=idx1)
    rh_out = pd.Series([60.0, 65.0], index=idx2)

    with pytest.raises(ValueError, match="index"):
        _ = vt.nocturnal_gain_by_angles(tilt_deg=90.0, t_out=t_out, rh_out=rh_out)


def test_nocturnal_gain_by_angles_rejects_invalid_tilt():
    idx = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
    rn_h = pd.Series([40.0, 50.0], index=idx)

    with pytest.raises(ValueError, match="tilt_deg"):
        _ = vt.nocturnal_gain_by_angles(tilt_deg=-1.0, rn_horizontal=rn_h)

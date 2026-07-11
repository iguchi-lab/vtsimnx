import pandas as pd
import pytest

import vtsimnx as vt


def test_solar_gain_by_angles_use_astro_runs():
    pytest.importorskip("astropy")
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=36.0,
        lon_deg=140.0,
        ghi=s_ig,
        use_astro=True,
        return_details=True,
    )

    assert "solar_azimuth_deg" in out.columns
    assert "solar_altitude_deg" in out.columns



import pandas as pd
import pytest

import vtsimnx as vt


def test_solar_gain_by_angles_rejects_ghi_plus_dhi_without_dni():
    idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1h")
    ghi = pd.Series([0.0, 100.0, 200.0], index=idx)
    dhi = pd.Series([0.0, 20.0, 30.0], index=idx)

    with pytest.raises(TypeError, match="ghi \\+ dhi"):
        _ = vt.solar_gain_by_angles(
            azimuth_deg=0.0,
            tilt_deg=90.0,
            ghi=ghi,
            dhi=dhi,
        )


def test_solar_gain_by_angles_requires_datetimeindex():
    idx = pd.Index([0, 1, 2])
    ghi = pd.Series([0.0, 100.0, 200.0], index=idx)

    with pytest.raises(TypeError, match="DatetimeIndex"):
        _ = vt.solar_gain_by_angles(
            azimuth_deg=0.0,
            tilt_deg=90.0,
            ghi=ghi,
        )


def test_solar_gain_by_angles_return_details_has_base_columns():
    idx = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1h")
    dni = pd.Series([0.0, 100.0, 200.0, 100.0], index=idx)
    dhi = pd.Series([0.0, 50.0, 60.0, 40.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        dni=dni,
        dhi=dhi,
        return_details=True,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(idx)
    assert {"法線面直達日射量 Ib", "水平面拡散日射量 Id", "太陽高度 hs", "太陽方位角 AZs"}.issubset(set(out.columns))

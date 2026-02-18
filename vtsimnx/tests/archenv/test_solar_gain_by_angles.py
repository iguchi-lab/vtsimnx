import numpy as np
import pandas as pd

import vtsimnx as vt


def test_solar_gain_by_angles_vertical_diffuse_parts_do_not_depend_on_azimuth():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    south = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )
    east = vt.solar_gain_by_angles(
        azimuth_deg=-90.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    np.testing.assert_allclose(
        south["水平面拡散日射量の拡散成分"].to_numpy(),
        east["水平面拡散日射量の拡散成分"].to_numpy(),
    )
    np.testing.assert_allclose(
        south["水平面拡散日射量の反射成分"].to_numpy(),
        east["水平面拡散日射量の反射成分"].to_numpy(),
    )



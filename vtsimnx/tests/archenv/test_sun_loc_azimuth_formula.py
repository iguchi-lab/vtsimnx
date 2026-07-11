import pandas as pd
import numpy as np

from vtsimnx.archenv.solar_position import sun_loc, cos_AZs


def test_sun_loc_cos_az_uses_declination_not_hour_angle():
    # 任意の数点で cos_AZs が定義式（赤緯delta_d）と一致することを確認する
    idx = pd.date_range("2026-08-15 06:00:00", periods=5, freq="3h")
    df = sun_loc(idx, lat=35.0, lon=139.0, td=0.0)

    expected = cos_AZs(
        df["sin_solar_altitude"].to_numpy(),
        35.0,
        df["solar_declination_deg"].to_numpy(),
        df["cos_solar_altitude"].to_numpy(),
    )

    np.testing.assert_allclose(
        df["cos_solar_azimuth"].to_numpy(),
        expected,
        rtol=0,
        atol=1e-12,
    )


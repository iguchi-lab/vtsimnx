import pandas as pd
import numpy as np

from vtsimnx.archenv.solar import sun_loc, cos_AZs


def test_sun_loc_cos_az_uses_declination_not_hour_angle():
    # 任意の数点で cos_AZs が定義式（赤緯delta_d）と一致することを確認する
    idx = pd.date_range("2026-08-15 06:00:00", periods=5, freq="3h")
    df = sun_loc(idx, lat=35.0, lon=139.0, td=0.0)

    expected = cos_AZs(
        df["太陽高度の正弦 sin_hs"].to_numpy(),
        35.0,
        df["太陽の赤緯 delta_d"].to_numpy(),
        df["太陽高度の余弦 cos_hs"].to_numpy(),
    )

    np.testing.assert_allclose(
        df["太陽方位角の余弦 cos_AZs"].to_numpy(),
        expected,
        rtol=0,
        atol=1e-12,
    )


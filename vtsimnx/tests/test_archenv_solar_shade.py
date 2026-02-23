import pandas as pd
import pytest

from vtsimnx.archenv.solar_shade import _normalize_shade_polygons, _shade_ratio_on_window


def test_normalize_shade_polygons_rejects_too_few_vertices():
    with pytest.raises(ValueError):
        _ = _normalize_shade_polygons([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])


@pytest.mark.parametrize(
    "window_width,window_height",
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_shade_ratio_on_window_rejects_non_positive_window_size(window_width, window_height):
    idx = pd.DatetimeIndex(["2026-06-21 12:00:00"])
    az = pd.Series([180.0], index=idx)
    hs = pd.Series([45.0], index=idx)
    shade = [[(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0)]]

    with pytest.raises(ValueError):
        _ = _shade_ratio_on_window(
            azs_deg=az,
            hs_deg=hs,
            surface_az_deg=0.0,
            surface_tilt_deg=90.0,
            window_width=window_width,
            window_height=window_height,
            shade_polygons_xyz=shade,
        )


def test_shade_ratio_on_window_is_zero_at_night():
    idx = pd.DatetimeIndex(["2026-01-01 00:00:00"])
    az = pd.Series([180.0], index=idx)
    hs = pd.Series([-10.0], index=idx)
    shade = [[(-2.0, 2.0, 0.0), (2.0, 2.0, 0.0), (2.0, -2.0, 0.0), (-2.0, -2.0, 0.0)]]

    eta = _shade_ratio_on_window(
        azs_deg=az,
        hs_deg=hs,
        surface_az_deg=0.0,
        surface_tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        shade_polygons_xyz=shade,
    )
    assert float(eta.iloc[0]) == 0.0

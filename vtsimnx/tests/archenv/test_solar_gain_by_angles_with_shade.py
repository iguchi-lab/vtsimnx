import numpy as np
import pandas as pd

import vtsimnx as vt


def test_solar_gain_by_angles_with_shade_default_returns_series():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        shade_coords=[(-5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (5.0, -5.0, 0.0), (-5.0, -5.0, 0.0)],
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
    )

    assert isinstance(out, pd.Series)
    assert out.name == "日射熱取得量"


def test_solar_gain_by_angles_with_shade_no_overlap_matches_base():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    base = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )
    shaded = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        # 窓から十分離した位置にシェードを置く（重ならない）
        shade_coords=[(10.0, 10.0, 1.0), (11.0, 10.0, 1.0), (11.0, 9.0, 1.0), (10.0, 9.0, 1.0)],
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    np.testing.assert_allclose(
        shaded["日向率(1-η)"].to_numpy(),
        1.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        shaded["直達日射量の面成分 Ib"].to_numpy(),
        base["直達日射量の面成分 Ib"].to_numpy(),
    )
    np.testing.assert_allclose(
        shaded["日射熱取得量"].to_numpy(),
        base["日射熱取得量"].to_numpy(),
    )


def test_solar_gain_by_angles_with_shade_full_cover_zeroes_direct():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        # 窓全体を覆う大きなポリゴン（窓面 z=0）
        shade_coords=[(-5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (5.0, -5.0, 0.0), (-5.0, -5.0, 0.0)],
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        glass=True,
        return_details=True,
    )

    assert np.allclose(out["被影率η"].to_numpy(), 1.0)
    assert np.allclose(out["日向率(1-η)"].to_numpy(), 0.0)
    assert np.allclose(out["直達日射量の面成分 Ib"].to_numpy(), 0.0)

    np.testing.assert_allclose(
        out["日射熱取得量"].to_numpy(),
        (out["水平面拡散日射量の拡散成分"] + out["水平面拡散日射量の反射成分"]).to_numpy(),
    )


def test_solar_gain_by_angles_with_shade_accepts_multiple_polygons():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        # 左半分 + 右半分を別ポリゴンで覆う（重なりなし）
        shade_coords=[
            [(-1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (-1.0, -1.0, 0.0)],
            [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (0.0, -1.0, 0.0)],
        ],
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    assert np.allclose(out["被影率η"].to_numpy(), 1.0)
    assert np.allclose(out["日向率(1-η)"].to_numpy(), 0.0)
    assert np.allclose(out["直達日射量の面成分 Ib"].to_numpy(), 0.0)


def test_solar_gain_by_angles_with_shade_overlapping_polygons_not_double_counted():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        # 2つのポリゴンは重なっており、単純和なら 1.5 になるが、
        # 和集合では窓全面(=1.0)で打ち止め
        shade_coords=[
            [(-1.0, 1.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 0.0), (-1.0, -1.0, 0.0)],
            [(-0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-0.5, -1.0, 0.0)],
        ],
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    assert np.allclose(out["被影率η"].to_numpy(), 1.0, atol=1e-9)
    assert np.allclose(out["日向率(1-η)"].to_numpy(), 0.0, atol=1e-9)


def test_solar_gain_by_angles_with_shade_top_center_origin_matches_center_origin():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    # 窓中心基準のポリゴン（窓上端付近の細い庇）
    shade_center = [(-1.0, 1.0, 0.5), (1.0, 1.0, 0.5), (1.0, 0.8, 0.5), (-1.0, 0.8, 0.5)]
    # 窓上端中央基準へ座標変換（y_top = y_center - H/2, H=2.0）
    shade_top_center = [(-1.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, -0.2, 0.5), (-1.0, -0.2, 0.5)]

    out_center = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        shade_coords=shade_center,
        shade_origin_mode="window_center",
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )
    out_top = vt.solar_gain_by_angles_with_shade(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        window_width=2.0,
        window_height=2.0,
        shade_coords=shade_top_center,
        shade_origin_mode="window_top_center",
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    np.testing.assert_allclose(out_center["被影率η"].to_numpy(), out_top["被影率η"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(out_center["日向率(1-η)"].to_numpy(), out_top["日向率(1-η)"].to_numpy(), atol=1e-9)

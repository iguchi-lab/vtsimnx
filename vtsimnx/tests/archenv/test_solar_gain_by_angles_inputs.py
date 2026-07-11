import numpy as np
import pandas as pd
import pytest

import vtsimnx as vt


def test_solar_gain_by_angles_default_returns_series():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        ghi=s_ig,
    )

    assert isinstance(out, pd.Series)
    assert out.name == "solar_gain"


def test_solar_gain_by_angles_accepts_ig_only():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        ghi=s_ig,
        return_details=True,
    )

    assert "solar_gain" in out.columns
    assert np.all(out["dhi"].to_numpy() >= -1e-9)


def test_solar_gain_by_angles_accepts_ig_and_ib_restores_id_nonnegative():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)
    s_ib = pd.Series([500.0, 500.0], index=idx)  # わざと過大

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        ghi=s_ig,
        dni=s_ib,
        return_details=True,
    )

    # Id >= 0 になる（IbはIG/sin(hs)で丸め）
    assert np.all(out["dhi"].to_numpy() >= -1e-9)


def test_solar_gain_by_angles_accepts_ib_and_id():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    assert "dni" in out.columns
    assert "dhi" in out.columns
    assert np.allclose(out["dni"].to_numpy(), 800.0)
    assert np.allclose(out["dhi"].to_numpy(), 100.0)


def test_solar_gain_by_angles_diffuse_only_zeroes_direct_terms():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=90.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        solar_mode="diffuse_only",
        return_details=True,
    )

    # 直達を 0 扱い
    assert np.allclose(out["beam_on_surface"].to_numpy(), 0.0)

    # 合計は拡散+反射のみになる
    np.testing.assert_allclose(
        out["solar_gain"].to_numpy(),
        (out["diffuse_sky_on_surface"] + out["diffuse_ground_reflected"]).to_numpy(),
    )


def test_solar_gain_by_angles_horizontal_equals_ghi():
    # 傾斜角=0（水平上向き）のとき
    #   直達面成分 = DNI*sin(hs)
    #   拡散面成分 = DHI
    #   地面反射 = 0
    # なので、合計=GHI(=DHI + DNI*sin(hs)) になることを確認する
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles(
        azimuth_deg=0.0,
        tilt_deg=0.0,
        lat_deg=35.0,
        lon_deg=139.0,
        dni=s_ib,
        dhi=s_id,
        return_details=True,
    )

    sin_hs = np.sin(np.radians(out["solar_altitude_deg"].to_numpy()))
    ghi_expected = s_id.to_numpy() + s_ib.to_numpy() * np.maximum(sin_hs, 0.0)
    np.testing.assert_allclose(out["solar_gain"].to_numpy(), ghi_expected, rtol=0, atol=1e-6)


def test_solar_gain_by_angles_rejects_ghi_plus_dhi_only():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    with pytest.raises(TypeError):
        _ = vt.solar_gain_by_angles(
            azimuth_deg=0.0,
            tilt_deg=90.0,
            lat_deg=35.0,
            lon_deg=139.0,
            ghi=s_ig,
            dhi=s_id,
            return_details=True,
        )


def test_solar_gain_by_angles_rejects_mismatched_indexes():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    idx_shift = pd.date_range("2026-06-21 13:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)
    s_ib = pd.Series([800.0, 800.0], index=idx_shift)

    with pytest.raises(ValueError):
        _ = vt.solar_gain_by_angles(
            azimuth_deg=0.0,
            tilt_deg=90.0,
            lat_deg=35.0,
            lon_deg=139.0,
            ghi=s_ig,
            dni=s_ib,
            return_details=True,
        )



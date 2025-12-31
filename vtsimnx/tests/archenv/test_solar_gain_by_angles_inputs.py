import numpy as np
import pandas as pd

import vtsimnx as vt


def test_solar_gain_by_angles_accepts_ig_only():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)

    out = vt.solar_gain_by_angles(
        方位角=0.0,
        傾斜角=90.0,
        緯度=35.0,
        経度=139.0,
        全天日射量=s_ig,
        名前="南面",
    )

    assert "日射熱取得量（南面）" in out.columns
    assert "日射熱取得量（南面ガラス）" in out.columns
    assert np.all(out["水平面拡散日射量 Id"].to_numpy() >= -1e-9)


def test_solar_gain_by_angles_accepts_ig_and_ib_restores_id_nonnegative():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx)
    s_ib = pd.Series([500.0, 500.0], index=idx)  # わざと過大

    out = vt.solar_gain_by_angles(
        方位角=0.0,
        傾斜角=90.0,
        緯度=35.0,
        経度=139.0,
        全天日射量=s_ig,
        法線面直達日射量=s_ib,
        名前="南面",
    )

    # Id >= 0 になる（IbはIG/sin(hs)で丸め）
    assert np.all(out["水平面拡散日射量 Id"].to_numpy() >= -1e-9)


def test_solar_gain_by_angles_accepts_ib_and_id():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles(
        方位角=0.0,
        傾斜角=90.0,
        緯度=35.0,
        経度=139.0,
        法線面直達日射量=s_ib,
        水平面拡散日射量=s_id,
        名前="南面",
    )

    assert "法線面直達日射量 Ib" in out.columns
    assert "水平面拡散日射量 Id" in out.columns
    assert np.allclose(out["法線面直達日射量 Ib"].to_numpy(), 800.0)
    assert np.allclose(out["水平面拡散日射量 Id"].to_numpy(), 100.0)


def test_solar_gain_by_angles_diffuse_only_zeroes_direct_terms():
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ib = pd.Series([800.0, 800.0], index=idx)
    s_id = pd.Series([100.0, 100.0], index=idx)

    out = vt.solar_gain_by_angles(
        方位角=0.0,
        傾斜角=90.0,
        緯度=35.0,
        経度=139.0,
        法線面直達日射量=s_ib,
        水平面拡散日射量=s_id,
        名前="南面",
        日射モード="diffuse_only",
    )

    # 直達を 0 扱い
    assert np.allclose(out["直達日射量の南面面成分 Ib"].to_numpy(), 0.0)
    assert np.allclose(out["直達日射量の南面面成分（ガラス） Ib_g"].to_numpy(), 0.0)

    # 合計は拡散+反射のみになる
    np.testing.assert_allclose(
        out["日射熱取得量（南面）"].to_numpy(),
        (out["水平面拡散日射量の拡散成分（南面）"] + out["水平面拡散日射量の反射成分（南面）"]).to_numpy(),
    )
    np.testing.assert_allclose(
        out["日射熱取得量（南面ガラス）"].to_numpy(),
        (
            out["水平面拡散日射量の拡散成分（ガラス）（南面）"]
            + out["水平面拡散日射量の反射成分（ガラス）（南面）"]
        ).to_numpy(),
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
        方位角=0.0,
        傾斜角=0.0,
        緯度=35.0,
        経度=139.0,
        法線面直達日射量=s_ib,
        水平面拡散日射量=s_id,
        名前="水平面",
    )

    sin_hs = np.sin(np.radians(out["太陽高度 hs"].to_numpy()))
    ghi_expected = s_id.to_numpy() + s_ib.to_numpy() * np.maximum(sin_hs, 0.0)
    np.testing.assert_allclose(out["日射熱取得量（水平面）"].to_numpy(), ghi_expected, rtol=0, atol=1e-6)



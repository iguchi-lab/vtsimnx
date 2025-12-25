import pandas as pd
import numpy as np

import vtsimnx as vt


def test_make_solar_accepts_direct_and_global_series():
    # 日中の時間帯を選ぶ（sin(hs)>0 になりやすい）
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    s_ig = pd.Series([200.0, 200.0], index=idx, name="水平面全天日射量")
    s_ib = pd.Series([500.0, 500.0], index=idx, name="直達日射量")  # 過大にして Id=0 / Ib が丸められることを期待

    df = vt.make_solar(全天日射量=s_ig, 法線面直達日射量=s_ib, 緯度=35.0, 経度=139.0)

    assert "水平面全天日射量" in df.columns
    assert "法線面直達日射量 Ib" in df.columns
    assert "水平面拡散日射量 Id" in df.columns
    assert "太陽高度の正弦 sin_hs" in df.columns

    # 復元式: Id = IG - Ib*sin(hs)（クリップ後なので 0 以上）
    id_calc = df["水平面全天日射量"] - df["法線面直達日射量 Ib"] * df["太陽高度の正弦 sin_hs"]
    assert np.all(df["水平面拡散日射量 Id"].to_numpy() >= -1e-9)
    np.testing.assert_allclose(
        df["水平面拡散日射量 Id"].to_numpy(),
        np.maximum(id_calc.to_numpy(), 0.0),
        rtol=0,
        atol=1e-6,
    )


def test_make_solar_dataframe_prefers_direct_and_global_over_erbs():
    # DataFrame 入力（HASP: 直達日射量 + 水平面全天日射量）を想定
    idx = pd.date_range("2026-06-21 12:00:00", periods=2, freq="1h")
    df_i = pd.DataFrame(
        {
            "直達日射量": [500.0, 500.0],
            "水平面全天日射量": [200.0, 200.0],
        },
        index=idx,
    )

    df = vt.make_solar(df_i, 緯度=35.0, 経度=139.0)

    # Erbs 経路だと「晴天指数 Kt」等が作られるが、この入力では Ib+IG 経路を優先するため存在しない
    assert "晴天指数 Kt" not in df.columns
    assert "水平面拡散日射量 Id" in df.columns


def test_make_solar_low_sun_cutoff_avoids_spike():
    # 日の出直後のような低高度を含みやすい時間を選ぶ（場所は適当）
    idx = pd.date_range("2026-01-01 06:00:00", periods=2, freq="1h")
    s_ig = pd.Series([50.0, 50.0], index=idx, name="水平面全天日射量")
    s_ib = pd.Series([800.0, 800.0], index=idx, name="直達日射量")

    df0 = vt.make_solar(全天日射量=s_ig, 法線面直達日射量=s_ib, 緯度=35.0, 経度=139.0, min_sun_alt_deg=0.0)
    df3 = vt.make_solar(全天日射量=s_ig, 法線面直達日射量=s_ib, 緯度=35.0, 経度=139.0, min_sun_alt_deg=3.0)

    # 3°カットオフでは、hs<=3° の区間があれば直達が 0 になり、過大な直達由来のスパイクを抑えられる
    low = df3["太陽高度 hs"] <= 3.0
    if low.any():
        assert np.all(df3.loc[low, "法線面直達日射量 Ib"].to_numpy() == 0.0)
        assert np.all(df3.loc[low, "日射熱取得量（南面）"].to_numpy() == 0.0)
    # 逆に、カットオフ無しより「絶対値が大きくなる」ことは起きにくい（単調性チェック）
    assert df3["日射熱取得量（南面）"].abs().max() <= df0["日射熱取得量（南面）"].abs().max() + 1e-6


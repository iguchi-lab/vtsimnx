import numpy as np
import pandas as pd

from vtsimnx.archenv.solar_separation import Ib, Id, sep_direct_diffuse


def _legacy_id(ig, kt):
    out = np.zeros(len(kt), dtype="float64")
    for i, k in enumerate(kt):
        if k <= 0.22:
            out[i] = ig[i] * (1 - 0.09 * k)
        elif (0.22 < k) and (k <= 0.80):
            out[i] = ig[i] * (
                0.9511
                - 0.1604 * k
                + 4.388 * (k ** 2)
                - 16.638 * (k ** 3)
                + 12.336 * (k ** 4)
            )
        elif 0.80 < k:
            out[i] = 0.365 * ig[i]
    return out


def _legacy_ib(ig, id_, alt, min_alt_deg=0.0):
    out = np.zeros(len(id_), dtype="float64")
    for i, idv in enumerate(id_):
        alt_i = float(alt[i])
        if alt_i <= float(min_alt_deg):
            out[i] = 0.0
            continue
        s = np.sin(np.radians(alt_i))
        if s <= 0.0:
            out[i] = 0.0
            continue
        out[i] = (ig[i] - idv) / s
        if (alt_i < 10.0) and (out[i] > ig[i]):
            out[i] = ig[i]
    return out


def test_id_vectorized_matches_legacy():
    ig = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype="float64")
    kt = np.array([0.0, 0.22, 0.5, 0.8, 1.1], dtype="float64")

    got = Id(ig, kt)
    exp = _legacy_id(ig, kt)

    assert np.allclose(got, exp)


def test_ib_vectorized_matches_legacy():
    ig = np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype="float64")
    id_ = np.array([0.0, 20.0, 50.0, 80.0, 100.0], dtype="float64")
    alt = np.array([-1.0, 0.0, 5.0, 15.0, 45.0], dtype="float64")

    got = Ib(ig, id_, alt, min_alt_deg=0.0)
    exp = _legacy_ib(ig, id_, alt, min_alt_deg=0.0)

    assert np.allclose(got, exp)


def test_sep_direct_diffuse_basic_shape_and_columns():
    idx = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1h")
    s_ig = pd.Series([0.0, 100.0, 200.0, 300.0], index=idx)
    s_hs = pd.Series([-5.0, 0.0, 10.0, 30.0], index=idx)

    df = sep_direct_diffuse(s_ig, s_hs)

    assert df.index.equals(idx)
    assert {"水平面全天日射量", "太陽高度", "晴天指数 Kt", "水平面拡散日射量 Id", "法線面直達日射量 Ib"}.issubset(
        set(df.columns)
    )

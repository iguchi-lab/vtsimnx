import numpy as np

from vtsimnx.archenv import (
    MJ_to_Wh,
    T,
    T_dash,
    Wh_to_MJ,
    capa_air,
    e,
    f,
    f_from_x,
    get_rho,
    hl,
    hs,
    ht,
    ps,
    x,
)


def test_unit_conversions_are_inverse():
    vals = np.array([0.0, 1.0, 100.0, 1234.5], dtype="float64")
    assert np.allclose(MJ_to_Wh(Wh_to_MJ(vals)), vals)


def test_basic_thermo_helpers_return_finite_values():
    t = np.array([0.0, 20.0, 30.0], dtype="float64")
    h = np.array([40.0, 50.0, 60.0], dtype="float64")

    assert np.all(np.isfinite(get_rho(t)))
    assert np.all(np.isfinite(T(t)))
    assert np.all(np.isfinite(T_dash(t)))
    assert np.all(np.isfinite(f(t)))
    assert np.all(np.isfinite(ps(t)))
    assert np.all(np.isfinite(e(t, h)))
    assert np.all(np.isfinite(x(t, h)))
    assert np.all(np.isfinite(hs(t)))
    assert np.all(np.isfinite(hl(t, h)))
    assert np.all(np.isfinite(ht(t, h)))


def test_enthalpy_relation_holds():
    t = 25.0
    h = 50.0
    assert ht(t, h) == hs(t) + hl(t, h)


def test_capa_air_and_f_from_x_scalar():
    assert capa_air(1.0) > 0.0
    assert f_from_x(0.0) == 0.0

import numpy as np

from vtsimnx.archenv import (
    MJ_to_Wh,
    to_kelvin,
    T_dash,
    Wh_to_MJ,
    capa_air,
    vapor_pressure_from_rh_pa,
    log10_saturation_vapor_pressure_hpa,
    vapor_pressure_from_humidity_ratio_gpkg_pa,
    air_density,
    latent_enthalpy_kjkg,
    sensible_enthalpy_kjkg,
    total_enthalpy_kjkg,
    saturation_vapor_pressure_pa,
    humidity_ratio_from_rh,
)


def test_unit_conversions_are_inverse():
    vals = np.array([0.0, 1.0, 100.0, 1234.5], dtype="float64")
    assert np.allclose(MJ_to_Wh(Wh_to_MJ(vals)), vals)


def test_basic_thermo_helpers_return_finite_values():
    t = np.array([0.0, 20.0, 30.0], dtype="float64")
    h = np.array([40.0, 50.0, 60.0], dtype="float64")

    assert np.all(np.isfinite(air_density(t)))
    assert np.all(np.isfinite(to_kelvin(t)))
    assert np.all(np.isfinite(T_dash(t)))
    assert np.all(np.isfinite(log10_saturation_vapor_pressure_hpa(t)))
    assert np.all(np.isfinite(saturation_vapor_pressure_pa(t)))
    assert np.all(np.isfinite(vapor_pressure_from_rh_pa(t, h)))
    assert np.all(np.isfinite(humidity_ratio_from_rh(t, h)))
    assert np.all(np.isfinite(sensible_enthalpy_kjkg(t)))
    assert np.all(np.isfinite(latent_enthalpy_kjkg(t, h)))
    assert np.all(np.isfinite(total_enthalpy_kjkg(t, h)))


def test_enthalpy_relation_holds():
    t = 25.0
    h = 50.0
    assert total_enthalpy_kjkg(t, h) == sensible_enthalpy_kjkg(t) + latent_enthalpy_kjkg(t, h)


def test_capa_air_and_f_from_x_scalar():
    assert capa_air(1.0) > 0.0
    assert vapor_pressure_from_humidity_ratio_gpkg_pa(0.0) == 0.0

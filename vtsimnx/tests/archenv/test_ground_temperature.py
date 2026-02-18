import numpy as np
import pandas as pd

import vtsimnx as vt


def test_ground_temperature_by_depth_constant_boundary_equals_deep_temp():
    idx = pd.date_range("2026-01-01 00:00:00", periods=48, freq="1h")
    t_out = pd.Series(10.0, index=idx)

    out = vt.ground_temperature_by_depth(
        depth_m=1.0,
        t_out=t_out,
        deep_layer_depth_m=10.0,
        deep_layer_temp_c=10.0,
    )

    assert isinstance(out, pd.Series)
    assert out.name == "地盤温度"
    np.testing.assert_allclose(out.to_numpy(), 10.0, atol=1e-9)


def test_ground_temperature_by_depth_multiple_depths_returns_dataframe():
    idx = pd.date_range("2026-01-01 00:00:00", periods=72, freq="1h")
    t_out = pd.Series(10.0 + 5.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, len(idx))), index=idx)

    out = vt.ground_temperature_by_depth(
        depth_m=[0.1, 1.0, 3.0],
        t_out=t_out,
        deep_layer_depth_m=10.0,
        deep_layer_temp_c=10.0,
    )

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["地盤温度_0.100m", "地盤温度_1.000m", "地盤温度_3.000m"]
    assert out.index.equals(idx)


def test_ground_temperature_by_depth_solar_term_raises_shallow_temperature():
    idx = pd.date_range("2026-01-01 00:00:00", periods=96, freq="1h")
    t_out = pd.Series(10.0, index=idx)
    solar = pd.Series(300.0, index=idx)

    base = vt.ground_temperature_by_depth(
        depth_m=0.1,
        t_out=t_out,
    )
    heated = vt.ground_temperature_by_depth(
        depth_m=0.1,
        t_out=t_out,
        solar_horizontal=solar,
        solar_to_surface_temp_coeff=0.01,
    )

    assert float(heated.mean()) > float(base.mean())


def test_ground_temperature_by_depth_spinup_reduces_initial_transient():
    idx = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h")
    t_out = pd.Series(10.0, index=idx)

    no_spinup = vt.ground_temperature_by_depth(
        depth_m=1.0,
        t_out=t_out,
        deep_layer_depth_m=10.0,
        deep_layer_temp_c=10.0,
        init_temp_c=0.0,
    )
    with_spinup = vt.ground_temperature_by_depth(
        depth_m=1.0,
        t_out=t_out,
        deep_layer_depth_m=10.0,
        deep_layer_temp_c=10.0,
        init_temp_c=0.0,
        spinup=True,
        spinup_cycles=5,
    )

    assert abs(float(with_spinup.iloc[0]) - 10.0) < abs(float(no_spinup.iloc[0]) - 10.0)


def test_ground_temperature_by_depth_spinup_cycle_validation():
    idx = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h")
    t_out = pd.Series(10.0, index=idx)

    try:
        _ = vt.ground_temperature_by_depth(
            depth_m=1.0,
            t_out=t_out,
            spinup=True,
            spinup_cycles=1,
        )
    except ValueError as e:
        assert "spinup_cycles" in str(e)
    else:
        raise AssertionError("Expected ValueError when spinup=True and spinup_cycles=1")


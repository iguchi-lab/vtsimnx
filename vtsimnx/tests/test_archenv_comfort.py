import pytest

import vtsimnx as vt
from vtsimnx.archenv import calc_C, calc_R, calc_RC


def test_calc_rc_components_consistency():
    r = calc_R(1.0, 25.0, 20.0)
    c = calc_C(1.0, 3.0, 25.0, 20.0)
    rc = calc_RC(1.0, 3.0, 25.0, 20.0, 20.0)
    assert rc == pytest.approx(r + c)


def test_calc_pmv_and_ppd_return_float_for_valid_inputs():
    pmv = vt.calc_PMV(Met=1.0, W=0.0, Clo=1.0, t_a=20, h_a=50, t_r=20, v_a=0.2)
    ppd = vt.calc_PPD(Met=1.0, W=0.0, Clo=1.0, t_a=20, h_a=50, t_r=20, v_a=0.2)

    assert isinstance(pmv, float)
    assert isinstance(ppd, float)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"Met": -0.1}, "Met"),
        ({"W": -0.1}, "W"),
        ({"Clo": -0.1}, "Clo"),
        ({"h_a": -1}, "h_a"),
        ({"h_a": 101}, "h_a"),
        ({"v_a": -0.1}, "v_a"),
    ],
)
def test_calc_pmv_rejects_invalid_inputs(kwargs, msg):
    base = dict(Met=1.0, W=0.0, Clo=1.0, t_a=20, h_a=50, t_r=20, v_a=0.2)
    base.update(kwargs)

    with pytest.raises(ValueError, match=msg):
        _ = vt.calc_PMV(**base)

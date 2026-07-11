"""快適性指標（PMV/PPD）とカビ指標。

PMV 反復は engine/archenv（C++）と揃える:
- max_iter = 100
- 適応的 omega
- 非収束時は t_cl を [-50, 100] に clamp
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .archenv import vapor_pressure_from_rh_pa

PMV_OMEGA_DEFAULT = 0.5
PMV_TOLERANCE = 1e-6
PMV_MAX_ITERATIONS = 100


def _calc_R(f_cl: float, t_cl: float, t_r: float) -> float:
    """放射熱伝達項（内部）。"""
    return 3.96e-8 * f_cl * (math.pow(t_cl + 273.0, 4) - math.pow(t_r + 273.0, 4))


def _calc_C(f_cl: float, h_c: float, t_cl: float, t_a: float) -> float:
    """対流熱伝達項（内部）。"""
    return f_cl * h_c * (t_cl - t_a)


def _calc_RC(f_cl: float, h_c: float, t_cl: float, t_a: float, t_r: float) -> float:
    """放射＋対流の合算項（内部）。"""
    return _calc_R(f_cl, t_cl, t_r) + _calc_C(f_cl, h_c, t_cl, t_a)


# 後方互換（非推奨）: 旧公開名
calc_R = _calc_R
calc_C = _calc_C
calc_RC = _calc_RC


def _validate_pmv_inputs(Met: float, W: float, Clo: float, h_a: float, v_a: float) -> None:
    if Met < 0:
        raise ValueError("Met は 0 以上である必要があります。")
    if Clo < 0:
        raise ValueError("Clo は 0 以上である必要があります。")
    if v_a < 0:
        raise ValueError("v_a は 0 以上である必要があります。")
    if not (0 <= h_a <= 100):
        raise ValueError("h_a は 0..100 [%] の範囲である必要があります。")
    if W < 0:
        raise ValueError("W は 0 以上である必要があります。")


def calc_PMV(
    Met: float = 1.0,
    W: float = 0.0,
    Clo: float = 1.0,
    t_a: float = 20.0,
    h_a: float = 50.0,
    t_r: float = 20.0,
    v_a: float = 0.2,
) -> float:
    """PMV（Predicted Mean Vote）。

    引数:
      Met: 代謝量 [met]
      W: 外部仕事 [W/m2]（代謝換算後の仕事項）
      Clo: 着衣量 [clo]
      t_a: 気温 [degC]
      h_a: 相対湿度 [%]
      t_r: 平均放射温度 [degC]
      v_a: 気流速度 [m/s]
    """
    _validate_pmv_inputs(Met, W, Clo, h_a, v_a)

    M, I_cl = Met * 58.2, Clo * 0.155
    f_cl = (1.00 + 1.290 * I_cl) if I_cl < 0.078 else (1.05 + 0.645 * I_cl)
    t_cl = t_a
    omega = PMV_OMEGA_DEFAULT
    converged = False

    for _ in range(PMV_MAX_ITERATIONS):
        h_c = max(2.38 * math.pow(abs(t_cl - t_a), 0.25), 12.1 * math.sqrt(v_a))
        new_t_cl = 35.7 - 0.028 * (M - W) - I_cl * _calc_RC(f_cl, h_c, t_cl, t_a, t_r)
        delta = new_t_cl - t_cl
        if abs(delta) <= PMV_TOLERANCE:
            t_cl = new_t_cl
            converged = True
            break
        if abs(delta) > 5.0:
            omega = max(omega * 0.5, 0.1)
        elif abs(delta) < 1.0:
            omega = min(omega * 1.1, 1.0)
        t_cl += delta * omega

    if not converged:
        t_cl = min(max(t_cl, -50.0), 100.0)
        warnings.warn(
            "calc_PMV did not converge within max_iter; t_cl was clamped.",
            RuntimeWarning,
        )

    h_c = max(2.38 * math.pow(abs(t_cl - t_a), 0.25), 12.1 * math.sqrt(v_a))
    e = float(vapor_pressure_from_rh_pa(t_a, h_a))
    E_d = 3.05e-3 * (5733 - 6.99 * (M - W) - e)
    E_s = 0.42 * ((M - W) - 58.15)
    E_re = 1.7e-5 * M * (5867 - e)
    C_re = 0.0014 * M * (34 - t_a)
    L = (M - W) - E_d - E_s - E_re - C_re - _calc_RC(f_cl, h_c, t_cl, t_a, t_r)
    return (0.303 * math.exp(-0.036 * M) + 0.028) * L


def calc_PPD(
    Met: float = 1.0,
    W: float = 0.0,
    Clo: float = 1.0,
    t_a: float = 20.0,
    h_a: float = 50.0,
    t_r: float = 20.0,
    v_a: float = 0.2,
) -> float:
    """PPD（Predicted Percentage of Dissatisfied）[%]。"""
    pmv = calc_PMV(Met, W, Clo, t_a, h_a, t_r, v_a)
    return 100.0 - 95.0 * math.exp(-0.03353 * math.pow(pmv, 4) - 0.2179 * math.pow(pmv, 2))


def calc_fungal_index(h: float, t: float) -> float:
    """Fungal Index（カビ指標）。

    引数:
      h: 相対湿度 [%]（C++ ``calc_fungal_index`` と同じ）。
         後方互換のため ``0 < h <= 1`` のときは割合とみなし DeprecationWarning を出す。
      t: 温度 [degC]
    """
    h_in = float(h)
    if 0.0 < h_in <= 1.0:
        warnings.warn(
            "calc_fungal_index: relative humidity in (0, 1] is treated as a fraction; "
            "prefer percent (0..100) to match C++ / docs.",
            DeprecationWarning,
            stacklevel=2,
        )
        h_frac = h_in
    else:
        h_frac = float(np.clip(h_in / 100.0, 0.0, 1.0))

    a = -0.3
    b = 0.685
    c1 = 0.95
    c2 = 0.07
    c3 = 25.0
    c4 = 7.2

    x = (h_frac - c1) / c2
    y = (float(t) - c3) / c4
    return float(187.25 * np.exp((((x**2) - 2 * a * x * y + (y**2)) ** b) / (2 * (a**2) - 2)) - 8.25)


__all__ = ["calc_PMV", "calc_PPD", "calc_fungal_index"]

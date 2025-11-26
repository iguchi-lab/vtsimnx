import math
import numpy as np

from .archenv import e


calc_R  = lambda f_cl, t_cl, t_r:               3.96e-8 * f_cl * (math.pow(t_cl + 273, 4) - math.pow(t_r + 273, 4))
calc_C  = lambda f_cl, h_c, t_cl, t_a:          f_cl * h_c * (t_cl - t_a)
calc_RC = lambda f_cl, h_c, t_cl, t_a, t_r:     calc_R(f_cl, t_cl, t_r) + calc_C(f_cl, h_c, t_cl, t_a)


def calc_PMV(Met = 1.0, W = 0.0, Clo = 1.0, t_a = 20, h_a = 50, t_r = 20, v_a = 0.2):
    """PMV（Predicted Mean Vote）の算出
    引数は ISO に準拠した代表値（Met, Clo, 風速など）。戻り値は PMV（無次元）。
    """
    M, I_cl      = Met * 58.2, Clo * 0.155
    f_cl         = (1.00 + 1.290 * I_cl) if I_cl < 0.078 else (1.05 + 0.645 * I_cl)
    t_cl         = t_a
    omega, error = 0.5, 1e12
    max_iter     = 200

    while abs(error) > 1e-6 and max_iter > 0:
        h_c      = max(2.38 * math.pow(abs(t_cl - t_a), 0.25), 12.1 * math.sqrt(v_a))
        New_t_cl = 35.7 - 0.028 * (M - W) - I_cl * calc_RC(f_cl, h_c, t_cl, t_a, t_r)
        error    = New_t_cl - t_cl
        t_cl     = t_cl + error * omega
        max_iter -= 1

    E_d  = 3.05e-3 * (5733 - 6.99 * (M - W) - e(t_a, h_a))
    E_s  = 0.42 * ((M - W) - 58.15)
    E_re = 1.7e-5 * M * (5867 - e(t_a, h_a))
    C_re = 0.0014 * M * (34 - t_a)
    L    = (M - W) - E_d - E_s - E_re - C_re - calc_RC(f_cl, h_c, t_cl, t_a, t_r)
    PMV  = (0.303 * math.exp(-0.036 * M) + 0.028) * L

    return PMV


def calc_PPD(Met = 1.0, W = 0.0, Clo = 1.0, t_a = 20, h_a = 50, t_r = 20, v_a = 0.2):
    """PPD（Predicted Percentage of Dissatisfied）の算出。戻り値は不満足者率 [%]。"""
    PMV = calc_PMV(Met, W, Clo, t_a, h_a, t_r, v_a)
    PPD = 100 - 95 * math.exp(-0.03353 * math.pow(PMV, 4) - 0.2179 * math.pow(PMV, 2))
    return PPD


def calc_fungal_index(h, t):
    """Fungal Index（カビ指標）の算出。相対湿度 h（0-1 or 0-100）と温度 t[℃]を想定。"""
    a  = - 0.3
    b  = 0.685
    c1 = 0.95
    c2 = 0.07
    c3 = 25
    c4 = 7.2

    x = (h - c1) / c2
    y = (t - c3) / c4

    FI = 187.25 * np.exp((((x ** 2) - 2 * a * x * y + (y ** 2)) ** b) /(2 * (a ** 2) - 2)) - 8.25

    return FI



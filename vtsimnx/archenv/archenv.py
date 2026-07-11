"""湿り空気・単位換算・共通定数（archenv コア）。

単位の原則:
- 角度: 度
- 温度: 摂氏（degC）
- 放射・日射: W/m2（必要に応じて関数内で換算）
- 風速: m/s
- 絶対湿度: kg/kg'（乾き空気基準）
- 相対湿度: %（0..100）※ calc_fungal_index も % を正とする
"""
from __future__ import annotations

from typing import Any

import numpy as np

############################################################################################################################
# 定数
############################################################################################################################
P_ATM = 101325.0  # 標準大気圧 [Pa]
Air_Cp = 1.005  # 空気の定圧比熱 [kJ/(kg·K)]
Vap_Cp = 1.846  # 水蒸気の定圧比熱 [kJ/(kg·K)]
Vap_L = 2501.1  # 水蒸気の蒸発潜熱 [kJ/kg]

# 夜間放射推算式で用いる工学定数（kcal 系慣習）。SI の Stefan–Boltzmann 5.67e-8 ではない。
SIGMA_NOCTURNAL = 4.88e-8
Sigma = SIGMA_NOCTURNAL  # 後方互換エイリアス

Solar_I = 1365  # 太陽定数 [W/m2]
AIR_DENSITY_REF_C = 20.0  # capa_air 既定の参照温度 [degC]


def capa_air(v: float | np.ndarray, t_c: float = AIR_DENSITY_REF_C) -> float | np.ndarray:
    """容積 v [m3] の空気の熱容量 [J/K]。

    密度は ``air_density(t_c)`` [kg/m3]、比熱は ``Air_Cp`` [kJ/(kg·K)]。
    """
    return v * Air_Cp * air_density(t_c) * 1000.0


############################################################################################################################
# 湿り空気の状態
############################################################################################################################
def air_density(t_c: float | np.ndarray) -> float | np.ndarray:
    """空気密度 ρ [kg/m3] の近似（理想気体）。t_c [degC]。"""
    return 353.25 / (t_c + 273.15)


def to_kelvin(t_c: float | np.ndarray) -> float | np.ndarray:
    """温度 [degC] → 絶対温度 [K]。"""
    return t_c + 273.15


def T_dash(t_c: float | np.ndarray) -> float | np.ndarray:
    """飽和水蒸気圧近似で用いる補助変数。"""
    return to_kelvin(100.0) / to_kelvin(t_c)


def Wh_to_MJ(v: float | np.ndarray) -> float | np.ndarray:
    """Wh → MJ。"""
    return v * 3.6 / 1000


def MJ_to_Wh(v: float | np.ndarray) -> float | np.ndarray:
    """MJ → Wh。"""
    return v * 1000 / 3.6


def log10_saturation_vapor_pressure_hpa(t_c: float | np.ndarray) -> float | np.ndarray:
    """飽和水蒸気圧 ps の対数近似核（log10(ps[hPa])）。t_c [degC]。"""
    t_dash = T_dash(t_c)
    return (
        -7.90298 * (t_dash - 1)
        + 5.02808 * np.log10(t_dash)
        - 1.3816e-7 * (np.power(10, 11.344 * (1 - 1 / t_dash)) - 1)
        + 8.1328e-3 * (np.power(10, -3.4919 * (t_dash - 1)) - 1)
        + np.log10(1013.246)
    )


def vapor_pressure_from_humidity_ratio_pa(x_kgkg: float | np.ndarray) -> float | np.ndarray:
    """絶対湿度（混合比）x [kg/kg'] から水蒸気圧 e [Pa] を求める。"""
    return (x_kgkg * P_ATM) / (0.622 + x_kgkg)


def vapor_pressure_from_humidity_ratio_gpkg_pa(x_gpkg: float | np.ndarray) -> float | np.ndarray:
    """絶対湿度（混合比）x [g/kg'] から水蒸気圧 e [Pa] を求める。"""
    return vapor_pressure_from_humidity_ratio_pa(x_gpkg / 1000.0)


def saturation_vapor_pressure_pa(t_c: float | np.ndarray) -> float | np.ndarray:
    """飽和水蒸気圧 ps [Pa]。t_c [degC]。"""
    return np.power(10, log10_saturation_vapor_pressure_hpa(t_c)) * 100


def vapor_pressure_from_rh_pa(t_c: float | np.ndarray, rh_pct: float | np.ndarray) -> float | np.ndarray:
    """相対湿度 RH [%] から水蒸気圧 e [Pa] を求める。"""
    return rh_pct / 100.0 * saturation_vapor_pressure_pa(t_c)


def humidity_ratio_from_rh(t_c: float | np.ndarray, rh_pct: float | np.ndarray) -> float | np.ndarray:
    """相対湿度 RH [%] から絶対湿度（混合比）x [kg/kg'] を求める。"""
    e_val = vapor_pressure_from_rh_pa(t_c, rh_pct)
    return 0.622 * (e_val / (P_ATM - e_val))


def relative_humidity_from_humidity_ratio(
    t_c: float | np.ndarray, x_kgkg: float | np.ndarray
) -> float | np.ndarray:
    """温度 t [degC] と絶対湿度 x [kg/kg'] から相対湿度 RH [%] を求める。"""
    e_val = vapor_pressure_from_humidity_ratio_pa(x_kgkg)
    p_sat = saturation_vapor_pressure_pa(t_c)
    rh = np.where(p_sat > 0, 100.0 * e_val / p_sat, 0.0)
    return np.clip(rh, 0.0, 100.0)


def sensible_enthalpy_kjkg(t_c: float | np.ndarray) -> float | np.ndarray:
    """顕熱エンタルピ [kJ/kg(DA)]。t_c [degC]。"""
    return Air_Cp * t_c


def latent_enthalpy_kjkg(t_c: float | np.ndarray, rh_pct: float | np.ndarray) -> float | np.ndarray:
    """潜熱エンタルピ [kJ/kg(DA)]。"""
    return humidity_ratio_from_rh(t_c, rh_pct) * (Vap_L + Vap_Cp * t_c)


def total_enthalpy_kjkg(t_c: float | np.ndarray, rh_pct: float | np.ndarray) -> float | np.ndarray:
    """全熱エンタルピ [kJ/kg(DA)]。"""
    return sensible_enthalpy_kjkg(t_c) + latent_enthalpy_kjkg(t_c, rh_pct)


############################################################################################################################
# 共通ヘルパー（パッケージ非公開。solar_position 等から import）
############################################################################################################################
def _alt_deg_from_sin(sin_alt: Any) -> Any:
    """sin(仰角) から仰角 [deg] を求める（数値誤差をクリップ）。"""
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


def _az_deg_from_sin_cos(sin_az: Any, cos_az: Any) -> Any:
    """sin/cos から方位角 [deg] を求める（arctan2 で象限処理込み）。"""
    return np.degrees(np.arctan2(sin_az, cos_az))


__all__ = [
    "P_ATM",
    "Air_Cp",
    "Vap_Cp",
    "Vap_L",
    "SIGMA_NOCTURNAL",
    "Sigma",
    "Solar_I",
    "AIR_DENSITY_REF_C",
    "capa_air",
    "air_density",
    "to_kelvin",
    "T_dash",
    "Wh_to_MJ",
    "MJ_to_Wh",
    "log10_saturation_vapor_pressure_hpa",
    "vapor_pressure_from_humidity_ratio_pa",
    "vapor_pressure_from_humidity_ratio_gpkg_pa",
    "saturation_vapor_pressure_pa",
    "vapor_pressure_from_rh_pa",
    "humidity_ratio_from_rh",
    "relative_humidity_from_humidity_ratio",
    "sensible_enthalpy_kjkg",
    "latent_enthalpy_kjkg",
    "total_enthalpy_kjkg",
]

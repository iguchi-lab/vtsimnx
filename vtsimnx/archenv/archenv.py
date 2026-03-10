"""archenv: 建築環境（気象・放射・快適性）の計算ユーティリティ群

主な機能:
- 風向・風圧の算出
- 夜間放射量の算出
- 太陽位置・方位別日射量の算出（簡易/astropy）
- PMV/PPD、Fungal Index の算出

単位の原則:
- 角度: 度
- 温度: 摂氏
- 放射・日射: W/m2（必要に応じて関数内で換算）
- 風速: m/s
"""
import numpy as np

############################################################################################################################
# 定数
############################################################################################################################
P_ATM    = 101325.0 # 標準大気圧 [Pa]
Air_Cp   = 1.005    # 空気の定圧比熱 [kJ/(kg·K)]
Vap_Cp   = 1.846    # 水蒸気の定圧比熱 [kJ/(kg·K)]
Vap_L    = 2501.1   # 水蒸気の蒸発潜熱 [kJ/kg]

Sigma    = 4.88e-8  # ステファン・ボルツマン定数
Solar_I  = 1365     # 太陽定数 [W/m2]

def capa_air(v):
    """容積 v[m3] の空気の熱容量 [J/K]。"""
    return v * Air_Cp * 1.2 * 1000

############################################################################################################################
#湿り空気の状態
############################################################################################################################
# 空気密度 ρ [kg/m3] の近似（理想気体：ρ ≒ 353.25 / (t+273.15)）
# t は気温 [℃]、353.25 は 101325/287 ≒ 353.25（標準大気圧/気体定数）
def air_density(t_c):
    """空気密度 ρ [kg/m3] の近似（理想気体）。"""
    return 353.25 / (t_c + 273.15)

# 絶対温度 T [K]
def to_kelvin(t_c):
    """温度 [℃] → 絶対温度 [K]。"""
    return t_c + 273.15

# 飽和水蒸気圧近似で用いる補助変数（100℃と t℃ の絶対温度比）
def T_dash(t_c):
    """飽和水蒸気圧近似で用いる補助変数。"""
    return to_kelvin(100.0) / to_kelvin(t_c)

# 単位換算
def Wh_to_MJ(v):
    """Wh → MJ。"""
    return v * 3.6 / 1000


def MJ_to_Wh(v):
    """MJ → Wh。"""
    return v * 1000 / 3.6

# 飽和水蒸気圧 ps の対数近似の核（Magnus/Tetens に類する式）
# log10_saturation_vapor_pressure_hpa(t_c) は log10(ps[hPa]) を返す
def log10_saturation_vapor_pressure_hpa(t_c):
    """飽和水蒸気圧 ps の対数近似核（log10(ps[hPa])）。"""
    t_dash = T_dash(t_c)
    return (
        -7.90298 * (t_dash - 1)
        + 5.02808 * np.log10(t_dash)
        - 1.3816e-7 * (np.power(10, 11.344 * (1 - 1 / t_dash)) - 1)
        + 8.1328e-3 * (np.power(10, -3.4919 * (t_dash - 1)) - 1)
        + np.log10(1013.246)
    )

# 絶対湿度 x [kg/kg'] → 水蒸気圧 e[Pa] への変換に使う補助
def vapor_pressure_from_humidity_ratio_pa(x_kgkg):
    """絶対湿度（混合比）x [kg/kg'] から水蒸気圧 e [Pa] を求める。"""
    return (x_kgkg * P_ATM) / (0.622 + x_kgkg)


def vapor_pressure_from_humidity_ratio_gpkg_pa(x_gpkg):
    """絶対湿度（混合比）x [g/kg'] から水蒸気圧 e [Pa] を求める。"""
    return vapor_pressure_from_humidity_ratio_pa(x_gpkg / 1000.0)

# 飽和水蒸気圧 ps [Pa]（hPa → Pa へ ×100）
def saturation_vapor_pressure_pa(t_c):
    """飽和水蒸気圧 ps [Pa]。"""
    return np.power(10, log10_saturation_vapor_pressure_hpa(t_c)) * 100

# 水蒸気圧 e [Pa]（相対湿度 h[%] から）
def vapor_pressure_from_rh_pa(t_c, rh_pct):
    """相対湿度 RH[%] から水蒸気圧 e [Pa] を求める。"""
    return rh_pct / 100.0 * saturation_vapor_pressure_pa(t_c)

# 絶対湿度 x [kg/kg']（水蒸気の混合比）
def humidity_ratio_from_rh(t_c, rh_pct):
    """相対湿度 RH[%] から絶対湿度（混合比）x [kg/kg'] を求める。"""
    e_val = vapor_pressure_from_rh_pa(t_c, rh_pct)
    return 0.622 * (e_val / (P_ATM - e_val))


def relative_humidity_from_humidity_ratio(t_c, x_kgkg):
    """温度 t [℃] と絶対湿度（混合比）x [kg/kg'] から相対湿度 RH[%] を求める。

    x = 0.622 * e / (P - e) の逆算で e を求め、RH = 100 * e / ps(t) とする。
    """
    e_val = vapor_pressure_from_humidity_ratio_pa(x_kgkg)
    p_sat = saturation_vapor_pressure_pa(t_c)
    rh = np.where(p_sat > 0, 100.0 * e_val / p_sat, 0.0)
    return np.clip(rh, 0.0, 100.0)

# 顕熱エンタルピ hs [kJ/kg]（Air_Cp は kJ/(kg·K) 前提）
def sensible_enthalpy_kjkg(t_c):
    """顕熱エンタルピ [kJ/kg(DA)]。"""
    return Air_Cp * t_c

# 潜熱エンタルピ hl [kJ/kg]：x·(潜熱 + 水蒸気顕熱)
def latent_enthalpy_kjkg(t_c, rh_pct):
    """潜熱エンタルピ [kJ/kg(DA)]。"""
    return humidity_ratio_from_rh(t_c, rh_pct) * (Vap_L + Vap_Cp * t_c)

# 全熱エンタルピ ht [kJ/kg]
def total_enthalpy_kjkg(t_c, rh_pct):
    """全熱エンタルピ [kJ/kg(DA)]。"""
    return sensible_enthalpy_kjkg(t_c) + latent_enthalpy_kjkg(t_c, rh_pct)

############################################################################################################################
# 共通ヘルパー
############################################################################################################################
def _alt_deg_from_sin(sin_alt):
    """sin(仰角) から仰角[deg]を求める（数値誤差をクリップ）"""
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


def _az_deg_from_sin_cos(sin_az, cos_az):
    """sin/ cos から方位角[deg]を求める（arctan2 で象限処理込み）"""
    return np.degrees(np.arctan2(sin_az, cos_az))

############################################################################################################################
#（以下は共通のみを残し、機能別は wind.py / nocturnal.py / solar.py / comfort.py に分離）
############################################################################################################################
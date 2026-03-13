#pragma once
#include "acmodel.h"                 // AirconSpec, COPResult, InputData を前提
#include "../../archenv/include/archenv.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

namespace acmodel {

class DuctCentralModel : public AirconSpec {
public:
    explicit DuctCentralModel(const nlohmann::json& spec);
    ~DuctCentralModel() override;

    COPResult estimateCOP(const std::string& mode, const InputData& inputdata) override;
    COPResult estimateCoolingCOP(const InputData& inputdata);
    COPResult estimateHeatingCOP(const InputData& inputdata);

    // 任意: インタフェース整合用（使わなくてもOK）
    double calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const override;
    double calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const override;

    std::string getModelName() const override { return "DUCT_CENTRAL"; }
    nlohmann::json getModelParameters() const override;

protected:
    bool isValidOperatingCondition(double outdoor_temp, double indoor_temp) const override;

private:
    // 物性・幾何定数（Python値に合わせる）
    static constexpr double C_P_AIR = 1006.0;     // [J/kg/K]
    static constexpr double RHO_AIR = 1.2;        // [kg/m3]
    static constexpr double L_WTR   = (2500.8 - 2.3668 * 27.0); // [kJ/kg]
    static constexpr double C_P_W   = 1.846;      // [kJ/kg/K]  ※式側で×1e3
    static constexpr double F_ATM   = 101325.0;   // [Pa]
    static constexpr double A_F_HEX = 0.23559;    // [m2]
    static constexpr double A_E_HEX = 6.396;      // [m2]

    // --- 内部ユーティリティ ---
    // JSON 仕様の取得（kW→W, m3/s→m3/hなど上位からの単位前提で変換）
    double getCapacity(const char* mode, const char* key) const;   // spec["Q"][mode][key] (kW)
    double getPower(const char* mode, const char* key) const;      // spec["P"][mode][key] (kW)
    double getFanPower(const char* mode, const char* key) const;   // spec["P_fan"][mode][key] (kW)
    double getVolume(const char* vwhere, const char* mode, const char* key) const; // spec[vwhere][mode][key] (m3/s)

    // 送風機電力[kW]
    static double calculateFanPower(double v_supply_m3h, double v_design_m3h, double p_fan_rated_W);

    // 伝熱係数
    static double latentHTCoeff(double v_flow_m3h);                                       // α'_c (式36b)
    static double sensibleHTCoeffCooling_mid_rtd(double v_flow_m3h);                      // α_c(mid/rtd) 固定 0.010376
    static double sensibleHTCoeffCooling_eval(double v_flow_m3h, double x_in_kgkg);       // α_c(eval)    X_in
    static double sensibleHTCoeffHeating(double v_flow_m3h);                              // α_c(暖房) (式35)

    // 表面温度
    static double coolingSurfaceTempEval(double theta_in, double theta_out,
                                         double v_flow_m3h, double alpha_c_Wm2K);         // (式32)
    static double heatingSurfaceTempEval(double theta_in, double theta_out,
                                         double v_flow_m3h, double alpha_c_Wm2K);         // (式31)

    // 冷媒温度等
    static double coolingCondensingTemp_mid_rtd(double theta_evp);                        // Tcnd(mid/rtd)
    static double coolingCondensingTemp_eval(double theta_ex, double theta_evp);          // Tcnd(eval)
    static double coolingSubcooling(double theta_cnd);                                    // SC(冷房)
    static double coolingSuperheating(double theta_cnd);                                  // SH(冷房)

    static double heatingCondensingTemp_from_surface(double theta_surface);               // Tcnd = clamp(surface, 65)
    static double heatingEvaporatingTemp_mid_rtd(double theta_cnd);                       // Tevp(mid/rtd)
    static double heatingEvaporatingTemp_eval(double theta_ex, double theta_cnd);         // Tevp(eval)
    static double heatingSubcooling(double theta_cnd);                                    // SC(暖房)
    static double heatingSuperheating(double theta_cnd);                                  // SH(暖房)

    // 効率比（区分線形）
    static double efficiencyRatioPiecewise(double q, double q_min, double q_mid, double q_rtd,
                                           double e_r_min, double e_r_mid, double e_r_rtd);

    // 飽和水蒸気圧・絶対湿度（Python式5a/5b準拠のローカル版）
    static double saturation_vapor_pressure_from_formula(double x_degC);
    static double X_saturated_from_surfaceT(double x_degC);

    // 中間/定格：q_CS(x)+q_CL(x)=target を満たす x（表面温度）を二分法で解く
    double solveCoolingSurfaceTempByBisect(double v_flow_m3h,
                                           double q_target_W,
                                           double theta_in_degC) const;

    // ログ
    mutable std::vector<std::string> preparationLogs_;
    mutable std::vector<std::string> calculationLogs_;
};

} // namespace acmodel

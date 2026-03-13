#pragma once
#include "acmodel.h"
#include "aircon_constants.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

namespace acmodel {

/**
 * 細井先生の潜熱評価式を用いたモデル（latentmodel の C++ 忠実移植版）
 */
class LatentEvaluateModel : public AirconSpec {
public:
    explicit LatentEvaluateModel(const nlohmann::json& spec);
    ~LatentEvaluateModel() override;

    COPResult estimateCOP(const std::string& mode, const InputData& inputdata) override;
    std::string getModelName() const override;
    nlohmann::json getModelParameters() const override;

    // 参考ユーティリティ（任意）
    double calculatePowerConsumption(double cooling_load, double outdoor_temp, double indoor_temp) const override;
    double calculateCoolingCapacity(double power_consumption, double outdoor_temp, double indoor_temp) const override;
    bool isValidOperatingCondition(double outdoor_temp, double indoor_temp) const override;

private:
    // 冷房・暖房の実体
    COPResult estimateCoolingCOP(const InputData& inputdata);
    COPResult estimateHeatingCOP(const InputData& inputdata);

    // 細井式と各種サブルーチン
    double getHeatExchangerFrontArea(double rated_capacity_W) const;
    double getHeatExchangerSurfaceArea(double rated_capacity_W) const;

    double calculateFanPowerByHosoi(double capacity_W) const;

    double calculateLatentSensibleHeatTransferCoeff(double v_flow_m3ph,
                                                    double x_in_kgpkg,
                                                    double a_f_hex_m2) const;
    double calculateLatentLatentHeatTransferCoeff(double v_flow_m3ph,
                                                  double a_f_hex_m2) const;

    double calculateCoolingSurfaceTemp(double theta_in_C, double theta_out_C,
                                       double v_flow_m3ph,
                                       double alpha_c_W_m2K,
                                       double a_e_hex_m2) const;

    double calculateHeatingSurfaceTemp(double theta_in_C, double theta_out_C,
                                       double v_flow_m3ph,
                                       double alpha_c_W_m2K,
                                       double a_e_hex_m2) const;

    double calculateCoolingCondensingTemp(double theta_ex_C, double theta_evp_C) const;
    double calculateHeatingEvaporatingTemp(double theta_ex_C, double theta_cnd_C) const;

    double calculateCoolingSubcooling(double theta_cnd_C) const;
    double calculateHeatingSubcooling(double theta_cnd_C) const;

    double calculateCoolingSuperheating(double theta_cnd_C) const;
    double calculateHeatingSuperheating(double theta_cnd_C) const;

    double calculateHeatingHeatTransferCoeff(double v_flow_m3ph,
                                             double a_f_hex_m2) const;

private:
    // ログ（任意）
    std::vector<std::string> preparationLogs_;
    std::vector<std::string> calculationLogs_;
};

} // namespace acmodel

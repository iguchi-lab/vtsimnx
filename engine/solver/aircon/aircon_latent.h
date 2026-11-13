#pragma once

#include "aircon/aircon_controller.h"

namespace aircon::latent {

struct LatentProcessResult {
    const char* regime = "unclassified"; // diagnostic branch, not a control state
    double sensibleHeatCapacity = 0.0; // [W]
    double latentHeatCapacity = 0.0;   // [W]
    double supplyX = 0.0;              // [kg/kg(DA)]
    // 除湿量 [kg/s]（正=凝縮で空気から除去）。mDot*(xIn-xSupply)+
    double condensationRateKgPerS = 0.0;
    double coilTemp = 0.0;             // [degC]
    double coilX = 0.0;                // [kg/kg(DA)]
    double supplyRhPercent = 0.0;      // [%]
    double bfRhPercentBeforeFallback = 0.0; // [%]
    bool rhExceeded = false;
    bool usedRh95Fallback = false;
};

double totalHeatCapacity(const LatentProcessResult& loads);

acmodel::InputData buildAcmodelInput(const AirconValidationData& validData,
                                     double sensibleHeatCapacity,
                                     double latentHeatCapacity,
                                     double airFlowRate);

// 冷房時: node.pre_rh [%] が有効で室絶対湿度が目標超なら、
// supplyX を absolute_humidity(T_in, pre_rh) に固定する理想除湿（能力制限なし）。
// latent_method より優先。未指定（NaN）または目標以下なら従来どおり。
LatentProcessResult estimateLatentProcess(const AirconValidationData& validData,
                                          OperationMode operationMode,
                                          double sensibleHeatCapacity,
                                          double airFlowRate,
                                          const VertexProperties& nodeProps,
                                          bool moistEnthalpyEnabled = false);

// 吹出湿度・除湿量を空調ノードへ反映（湿気移流境界と診断の正本）。
// 固定境界を変更した場合は必ず再計算。許容内なら解いた境界値を保持する。
bool applySupplyHumidityToAirconNode(ThermalNetwork& thermalNetwork,
                                     const std::string& airconKey,
                                     const LatentProcessResult& loads,
                                     double humidityAbsTol = 1e-9);

// OFF（または送風のみ）: 除湿量を 0 にする。湿度は室と同時に求解する。
// 固定境界から未知数へ変わる場合 true（外側ループ再計算要求用）。
bool applyPassthroughHumidityToAirconNode(ThermalNetwork& thermalNetwork,
                                          const std::string& airconKey,
                                          double humidityAbsTol = 1e-9);

} // namespace aircon::latent

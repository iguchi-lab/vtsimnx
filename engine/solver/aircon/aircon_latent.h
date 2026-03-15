#pragma once

#include "aircon/aircon_controller.h"

namespace aircon::latent {

struct LatentProcessResult {
    double sensibleHeatCapacity = 0.0; // [W]
    double latentHeatCapacity = 0.0;   // [W]
    double supplyX = 0.0;              // [kg/kg(DA)]
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

LatentProcessResult estimateLatentProcess(const AirconValidationData& validData,
                                          OperationMode operationMode,
                                          double sensibleHeatCapacity,
                                          double airFlowRate,
                                          const VertexProperties& nodeProps);

} // namespace aircon::latent

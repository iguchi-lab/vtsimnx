#pragma once

#include "aircon/aircon_operation_mode.h"
#include "network/ventilation_network.h"

#include <optional>
#include <string>

namespace aircon::airflow {

bool isDuctCentralModel(const VertexProperties& nodeProps);

bool updateFixedFlowEdgeByNodePair(VentilationNetwork& ventNetwork,
                                   const std::string& fromNode,
                                   const std::string& toNode,
                                   double targetFlowM3s,
                                   double flowTolM3s);

std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW);

} // namespace aircon::airflow

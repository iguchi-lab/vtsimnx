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

// 還気（in_node -> 空調）と吹出（空調 -> out）の fixed_flow を同じ風量に揃える。
// 向きはビルダーと同じく、吸込は空調へ入り、吹出は空調から出る（どちらも枝方向に +q）。
// in == out のループでも吹出を逆符号にしない。変更があれば true。
bool updateDuctCentralCircuitFixedFlows(VentilationNetwork& ventNetwork,
                                        const std::string& inNode,
                                        const std::string& airconNode,
                                        double targetFlowM3s,
                                        double flowTolM3s);

std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW);

} // namespace aircon::airflow

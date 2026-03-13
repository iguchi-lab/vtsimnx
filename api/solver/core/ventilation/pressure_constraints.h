#pragma once

#include "../../vtsim_solver.h"
#include <ostream>
#include <unordered_map>
#include <vector>

namespace ceres { class CostFunction; }

// Ceresの CostFunction 実装は `.cpp` へ分離（ヘッダ依存・ビルド時間削減）。
namespace PressureConstraints {

ceres::CostFunction* createFlowBalanceConstraint(
    const std::string& nodeName,
    const Graph& graph,
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile);

ceres::CostFunction* createGroupFlowBalanceConstraint(
    const std::vector<Vertex>& groupVertices,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    size_t numParameters,
    std::ostream& logFile);

ceres::CostFunction* createSoftAnchorConstraint(
    size_t parameterIndex,
    double targetPressure,
    double weight,
    size_t numParameters);

} // namespace PressureConstraints

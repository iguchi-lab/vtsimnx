#pragma once

#include "../../vtsim_solver.h"
#include <ostream>
#include <unordered_map>
#include <vector>

namespace ceres { class CostFunction; class Problem; }

// Ceresの CostFunction 実装は `.cpp` へ分離（ヘッダ依存・ビルド時間削減）。
namespace PressureConstraints {

// densityByVertex: 圧力求解開始時に埋めた頂点配列。nullptr なら温度から都度計算。
ceres::CostFunction* createFlowBalanceConstraint(
    Vertex nodeVertex,
    const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex,
    size_t numParameters,
    std::ostream& logFile);

// 互換: nodeName から Vertex を解決
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
    const std::vector<double>* densityByVertex,
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

// Scalar pressure blocks: each balance depends only on incident unknowns.
// The Problem owns the cost; graph state and pressure storage must stay stable
// during Solve. Constant fixed-flow balances still get a zero-parameter residual.
void addFlowBalanceConstraint(
    ceres::Problem& problem, Vertex nodeVertex, const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex, const std::vector<double>& pressures);

void addGroupFlowBalanceConstraint(
    ceres::Problem& problem, const std::vector<Vertex>& groupVertices, const Graph& graph,
    const std::vector<int>& vertexToParameterIndexVec,
    const std::vector<std::vector<Edge>>& incidentEdgesByVertex,
    const std::vector<double>* densityByVertex, const std::vector<double>& pressures);

} // namespace PressureConstraints

#pragma once

// Internal definitions for PressureSolver private nested structs.
// This header is intentionally not included from `pressure_solver.h`
// to keep public compile-time dependencies low.

#include "core/ventilation/pressure_solver_impl.h"
#include <ceres/ceres.h>

#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct PressureSolver::Impl::TrialResult {
    bool converged = false;
    double usedTolerance = 0.0;
};

struct PressureSolver::Impl::SolverSetup {
    std::vector<std::string> nodeNames;
    std::vector<double> pressures;
    std::map<Vertex, size_t> vertexToParameterIndex;
    // 高速化用: Vertex(=0..V-1) -> param index（無ければ -1）
    std::vector<int> vertexToParameterIndexVec;
    // 高速化用: incident edges（全エッジ走査を避ける）
    std::vector<std::vector<Edge>> incidentEdgesByVertex;
};

struct PressureSolver::Impl::StageAMapping {
    std::map<int, size_t> groupToParamIndex;
    std::map<Vertex, size_t> vertexToParamIndex;
    // 高速化用: Vertex -> param index（無ければ -1）
    std::vector<int> vertexToParamIndexVec;
    std::vector<std::string> nodeNames;
    size_t parameterCount = 0;
};

struct PressureSolver::Impl::StageBSetup {
    std::map<Vertex, size_t> vertexToParamIndex;
    // 高速化用: Vertex -> param index（無ければ -1）
    std::vector<int> vertexToParamIndexVec;
    std::vector<std::string> nodeNames;
    std::vector<double> pressures;
};

struct PressureSolver::Impl::SupernodePartition {
    std::vector<Vertex> vertices;
    std::map<Vertex, int> v2i;
    std::vector<int> groupOfVertex;
    std::vector<std::vector<Edge>> incidentEdgesByVertex;
    size_t highEdgeCount = 0;
    size_t condensedNodeCount = 0;
};

struct PressureSolver::Impl::StageASolveResult {
    StageAMapping mapping;
    std::vector<double> pressures;
    PressureMap pressureMap;
    ceres::Solver::Summary summary;
    bool ok = false;
    int superCount = 0;
    double anchorTargetPressure = 0.0;
    bool hasAnchorTarget = false;
};

struct PressureSolver::Impl::InterfaceFreezeResult {
    size_t fixedFlowCount = 0;
    size_t alreadyFixed = 0;
    bool skipped = false;
    // convertToFixedFlow したエッジと、そのときの固定流量 Q_fixed [m³/s]
    std::vector<std::pair<Edge, double>> frozenFlows;
};

struct PressureSolver::Impl::StageBSolveResult {
    StageBSetup setup;
    PressureMap pressureMap;
    ceres::Solver::Summary summary;
    bool ok = false;
};

struct PressureSolver::Impl::FallbackOuterState {
    double lastCostOuter = std::numeric_limits<double>::infinity();
    double lastNetworkCostOuter = std::numeric_limits<double>::infinity();
    PressureMap prevPressureMapFB;
    PressureMap finalPressureMapFB;
    FlowRateMap finalFlowRatesFB;
    FlowBalanceMap finalBalanceFB;
    bool finalHaveSolution = false;
};


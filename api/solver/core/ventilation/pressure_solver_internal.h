#pragma once

// Internal definitions for PressureSolver private nested structs.
// This header is intentionally not included from `pressure_solver.h`
// to keep public compile-time dependencies low.

#include "core/ventilation/pressure_solver.h"

#include <map>
#include <string>
#include <vector>

struct PressureSolver::TrialResult {
    bool converged = false;
    double usedTolerance = 0.0;
};

struct PressureSolver::SolverSetup {
    std::vector<std::string> nodeNames;
    std::vector<double> pressures;
    std::map<Vertex, size_t> vertexToParameterIndex;
    // 高速化用: Vertex(=0..V-1) -> param index（無ければ -1）
    std::vector<int> vertexToParameterIndexVec;
    // 高速化用: incident edges（全エッジ走査を避ける）
    std::vector<std::vector<Edge>> incidentEdgesByVertex;
};

struct PressureSolver::StageAMapping {
    std::map<int, size_t> groupToParamIndex;
    std::map<Vertex, size_t> vertexToParamIndex;
    // 高速化用: Vertex -> param index（無ければ -1）
    std::vector<int> vertexToParamIndexVec;
    std::vector<std::string> nodeNames;
    size_t parameterCount = 0;
};

struct PressureSolver::StageBSetup {
    std::map<Vertex, size_t> vertexToParamIndex;
    // 高速化用: Vertex -> param index（無ければ -1）
    std::vector<int> vertexToParamIndexVec;
    std::vector<std::string> nodeNames;
    std::vector<double> pressures;
};



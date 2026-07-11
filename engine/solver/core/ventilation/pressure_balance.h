#pragma once

#include "../../vtsim_solver.h"

#include <cmath>
#include <cstddef>

namespace ventilation {

// 換気質量収支の物理指標。
// ventilationTolerance は「ノード流量収支の最大絶対値 [kg/s]」として解釈する。
struct BalanceMetrics {
    double maxAbs = 0.0;
    double l1 = 0.0;
    double l2 = 0.0;
    double rmse = 0.0;
    std::size_t nodeCount = 0;
};

inline BalanceMetrics computeBalanceMetrics(const FlowBalanceMap& balance) {
    BalanceMetrics m;
    double sumsq = 0.0;
    for (const auto& kv : balance) {
        const double a = std::abs(kv.second);
        m.maxAbs = std::max(m.maxAbs, a);
        m.l1 += a;
        sumsq += kv.second * kv.second;
        ++m.nodeCount;
    }
    m.l2 = std::sqrt(sumsq);
    m.rmse = (m.nodeCount > 0) ? std::sqrt(sumsq / static_cast<double>(m.nodeCount)) : 0.0;
    return m;
}

inline bool acceptMassBalance(const BalanceMetrics& metrics, double massBalanceMaxAbs) {
    if (!(massBalanceMaxAbs > 0.0)) {
        return false;
    }
    return metrics.maxAbs <= massBalanceMaxAbs;
}

// Ceres 停止条件と物理合否を分離する内部設定。
struct PressureSolverTolerances {
    double massBalanceMaxAbs = 1e-6;      // [kg/s] max |node balance|
    double ceresFunctionRelative = 1e-12; // Ceres function_tolerance（相対）
    double ceresParameter = 1e-12;        // Ceres parameter_tolerance
    double ceresGradient = 1e-10;         // Ceres gradient_tolerance
};

inline PressureSolverTolerances makePressureSolverTolerances(const SimulationConstants& constants) {
    PressureSolverTolerances t;
    t.massBalanceMaxAbs = constants.ventilationTolerance;
    return t;
}

} // namespace ventilation

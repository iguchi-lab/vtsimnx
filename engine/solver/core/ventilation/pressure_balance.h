#pragma once

#include "../../vtsim_solver.h"

#include <cmath>
#include <cstddef>
#include <string>

namespace ventilation {

// 換気体積流量収支の物理指標。
// ventilationTolerance は「計算圧力ノード（calc_p）体積流量収支の最大絶対値 [m³/s]」として解釈する。
// （質量流量ではなく体積流量。湿度移流のみ後段で ρ を掛けて [kg/s] に変換する。）
struct BalanceMetrics {
    double maxAbs = 0.0;
    double l1 = 0.0;
    double l2 = 0.0;
    double rmse = 0.0;
    std::size_t nodeCount = 0;
    // calc_p ノードがすべて balance に存在する（欠落なし）
    bool complete = true;
    // 集計に使った収支値がすべて有限
    bool finite = true;
};

inline BalanceMetrics computeBalanceMetrics(const FlowBalanceMap& balance) {
    BalanceMetrics m;
    double sumsq = 0.0;
    for (const auto& kv : balance) {
        if (!std::isfinite(kv.second)) {
            m.finite = false;
            ++m.nodeCount;
            continue;
        }
        const double a = std::abs(kv.second);
        m.maxAbs = std::max(m.maxAbs, a);
        m.l1 += a;
        sumsq += kv.second * kv.second;
        ++m.nodeCount;
    }
    m.l2 = std::sqrt(sumsq);
    m.rmse = (m.nodeCount > 0 && m.finite)
                 ? std::sqrt(sumsq / static_cast<double>(m.nodeCount))
                 : 0.0;
    return m;
}

// 固定圧力境界（calc_p=false）は外部リザーバなので収支ゼロを要求しない。
// calc_p ノードの balance 欠落は complete=false（スキップして合格にしない）。
inline BalanceMetrics computePressureUnknownBalanceMetrics(
        const FlowBalanceMap& balance,
        const Graph& graph) {
    BalanceMetrics m;
    double sumsq = 0.0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        if (!graph[v].calc_p) {
            continue;
        }
        auto it = balance.find(graph[v].key);
        if (it == balance.end()) {
            m.complete = false;
            continue;
        }
        if (!std::isfinite(it->second)) {
            m.finite = false;
            ++m.nodeCount;
            continue;
        }
        const double a = std::abs(it->second);
        m.maxAbs = std::max(m.maxAbs, a);
        m.l1 += a;
        sumsq += it->second * it->second;
        ++m.nodeCount;
    }
    m.l2 = std::sqrt(sumsq);
    m.rmse = (m.nodeCount > 0 && m.finite && m.complete)
                 ? std::sqrt(sumsq / static_cast<double>(m.nodeCount))
                 : 0.0;
    return m;
}

inline bool acceptMassBalance(const BalanceMetrics& metrics, double massBalanceMaxAbs) {
    if (!(massBalanceMaxAbs > 0.0)) {
        return false;
    }
    return metrics.nodeCount > 0
        && metrics.complete
        && metrics.finite
        && std::isfinite(metrics.maxAbs)
        && metrics.maxAbs <= massBalanceMaxAbs;
}

// Stage A / interface freeze が成功したときだけ Stage B へ進む。
inline bool canProceedToFallbackStageB(bool stageAOk, bool interfaceFreezeSkipped) {
    return stageAOk && !interfaceFreezeSkipped;
}

// log10 コンダクタンスの2クラスタ中心が、明確な強弱分離を持つか。
// minLogSep: cHigh-cLow の最小差（log10）、minRatio: 線形空間での G_high/G_low。
inline bool hasClearConductanceSeparation(
        double cLowLog10,
        double cHighLog10,
        double minLogSep = 1.0,
        double minRatio = 10.0) {
    if (!(std::isfinite(cLowLog10) && std::isfinite(cHighLog10))) {
        return false;
    }
    if (cHighLog10 < cLowLog10) {
        std::swap(cLowLog10, cHighLog10);
    }
    const double logSep = cHighLog10 - cLowLog10;
    if (!(logSep >= minLogSep)) {
        return false;
    }
    const double ratio = std::pow(10.0, logSep);
    return ratio >= minRatio;
}

// 固定流量化エッジについて、復元後の元特性流量と固定値の差。
struct InterfaceFlowConsistency {
    double maxAbs = 0.0;
    std::size_t edgeCount = 0;
    bool finite = true;
    bool ok = false;
};

inline bool acceptInterfaceFlowConsistency(
        const InterfaceFlowConsistency& metrics,
        double interfaceFlowMaxAbs) {
    if (!(interfaceFlowMaxAbs > 0.0)) {
        return false;
    }
    return metrics.ok
        && metrics.finite
        && std::isfinite(metrics.maxAbs)
        && metrics.maxAbs <= interfaceFlowMaxAbs;
}

// primary / fallback 共通の解評価結果（出力用 balance は全ノード）。
struct PressureSolutionEvaluation {
    FlowRateMap flows;
    FlowBalanceMap allNodeBalances;
    BalanceMetrics solvedNodeMetrics;
    bool flowOk = false;
    bool accepted = false;
    std::string detail;
};

// Ceres 停止条件と物理合否を分離する内部設定。
struct PressureSolverTolerances {
    double massBalanceMaxAbs = 1e-6;      // [m³/s] max |calc_p node volume-flow balance|
    double interfaceFlowMaxAbs = 1e-6;    // [m³/s] max |Q_original - Q_fixed|
    double ceresFunctionRelative = 1e-12; // Ceres function_tolerance（相対）
    double ceresParameter = 1e-12;        // Ceres parameter_tolerance
    double ceresGradient = 1e-10;         // Ceres gradient_tolerance
};

inline PressureSolverTolerances makePressureSolverTolerances(const SimulationConstants& constants) {
    PressureSolverTolerances t;
    t.massBalanceMaxAbs = constants.ventilationTolerance;
    t.interfaceFlowMaxAbs = constants.ventilationTolerance;
    return t;
}

} // namespace ventilation

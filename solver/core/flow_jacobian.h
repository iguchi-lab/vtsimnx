#pragma once

#include "../vtsim_solver.h"
#include "core/flow_calculation.h"
#include <cmath>

// =============================================================================
// FlowJacobian - 風量計算のヤコビアン（d(flow)/d(dp)）の共通関数群
// =============================================================================
//
// dp は「(source_total_pressure - target_total_pressure)」を想定。
// 返すのは dQ/d(dp)（= dQ/dp_source、dQ/dp_target は符号で分配する）。

namespace FlowJacobian {
// simple_openingのヤコビアン（解析）
inline double calcSimpleOpeningJacobian(double dp, const EdgeProperties& edgeData) {
    const double eps = archenv::TOLERANCE_SMALL;
    const double abs_dp = std::abs(dp);
    const double K = edgeData.alpha * edgeData.area * std::sqrt(2.0 / archenv::DENSITY_DRY_AIR);
    if (abs_dp >= eps) {
        // sign^2 = 1 なので符号は不要
        return 0.5 * K / std::sqrt(abs_dp);
    }
    // 小差圧では線形近似の傾き
    return K * std::sqrt(eps) / eps;
}

// gapのヤコビアン（解析）
inline double calcGapJacobian(double dp, const EdgeProperties& edgeData) {
    const double eps = archenv::TOLERANCE_SMALL;
    const double abs_dp = std::abs(dp);
    double n = edgeData.n;
    if (n == 0.0) n = 1.0;
    const double a = edgeData.a;
    if (abs_dp >= eps) {
        // sign^2 = 1 なので符号は不要
        return a * (1.0 / n) * std::pow(abs_dp, 1.0 / n - 1.0);
    }
    return a * std::pow(eps, 1.0 / n - 1.0);
}
} // namespace FlowJacobian

namespace FanJacobian {
// fanのヤコビアン（区分線形の解析）
inline double calcFanJacobian(double dp, const EdgeProperties& edgeData) {
    const double dp_fan = -dp;  // ファンは逆方向の圧力差
    const double p_max = edgeData.p_max;
    const double p1 = edgeData.p1;
    const double q1 = edgeData.q1;
    const double q_max = edgeData.q_max;

    // スムージング無しの明確な閾値で区分
    if (dp_fan >= p_max) {
        return 0.0; // Q=0
    } else if (dp_fan >= p1) {
        if (p1 == p_max) return 0.0;
        // Q = q1 * (dp_fan - p_max)/(p1 - p_max)
        // dQ/dp = dQ/d(dp_fan)*d(dp_fan)/dp = (q1/(p1-p_max))*(-1)
        return -q1 / (p1 - p_max);
    } else if (dp_fan >= 0.0) {
        if (p1 == 0.0) return 0.0;
        // Q = q1 + (q_max-q1)*(dp_fan - p1)/(-p1)
        // dQ/dp = ((q_max-q1)/(-p1))*(-1) = (q_max-q1)/p1
        return (q_max - q1) / p1;
    } else {
        return 0.0; // Q=q_max（定数）
    }
}
} // namespace FanJacobian

namespace FlowJacobianCommon {
// 統一されたヤコビアン計算関数（d(flow)/d(dp)）
inline double calculateJacobian(double dp, const EdgeProperties& edgeData) {
    if (edgeData.type == "fan") {
        return FanJacobian::calcFanJacobian(dp, edgeData);
    } else if (edgeData.type == "simple_opening") {
        return FlowJacobian::calcSimpleOpeningJacobian(dp, edgeData);
    } else if (edgeData.type == "gap") {
        return FlowJacobian::calcGapJacobian(dp, edgeData);
    } else if (edgeData.type == "fixed_flow") {
        return 0.0;
    }

    // その他は互換性のため数値微分（未知タイプが来た時だけ）
    const double eps = 1e-7;
    const double q_plus = FlowCalculation::calculateUnifiedFlow(dp + eps, edgeData);
    const double q_minus = FlowCalculation::calculateUnifiedFlow(dp - eps, edgeData);
    double dQdp = (q_plus - q_minus) / (2.0 * eps);
    if (!std::isfinite(dQdp)) dQdp = 0.0;
    return dQdp;
}
} // namespace FlowJacobianCommon



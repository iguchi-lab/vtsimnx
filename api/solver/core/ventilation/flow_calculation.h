#pragma once

#include "../../vtsim_solver.h"
#include "../../archenv/include/archenv.h"
#include <cmath>

// =============================================================================
// FlowCalculation - 風量計算の共通関数群（double専用）
// =============================================================================

namespace FlowCalculation {

// 開口部の風量計算（simple_opening）
inline double calcSimpleOpeningFlow(double dp, const EdgeProperties& edgeData) {
    const double alpha = edgeData.alpha;
    const double area  = edgeData.area;
    const double eps   = archenv::TOLERANCE_SMALL;
    const double abs_dp = std::abs(dp);
    const double K = alpha * area * std::sqrt(2.0 / archenv::DENSITY_DRY_AIR);
    if (abs_dp >= eps) {
        const double sign = (dp >= 0.0) ? 1.0 : -1.0;
        return sign * K * std::sqrt(abs_dp);
    }
    // 小差圧は線形近似
    return K * std::sqrt(eps) * (dp / eps);
}

// 隙間の風量計算（gap）
inline double calcGapFlow(double dp, const EdgeProperties& edgeData) {
    const double a  = edgeData.a;
    double n = edgeData.n;
    if (n == 0.0) n = 1.0;
    const double eps = archenv::TOLERANCE_SMALL;
    const double abs_dp = std::abs(dp);
    if (abs_dp >= eps) {
        const double sign = (dp >= 0.0) ? 1.0 : -1.0;
        return sign * a * std::pow(abs_dp, 1.0 / n);
    }
    // 小差圧は線形近似
    return a * std::pow(eps, 1.0 / n - 1.0) * dp;
}

// ファンの風量計算（fan）
inline double calcFanFlow(double dp, const EdgeProperties& edgeData) {
    const double q_max = edgeData.q_max;
    const double p_max = edgeData.p_max;
    const double p1 = edgeData.p1;
    const double q1 = edgeData.q1;

    const double dp_fan = -dp;
    const double tolerance = archenv::TOLERANCE_MEDIUM;

    auto smoothCondition = [tolerance](double x, double threshold) -> double {
        const double diff = x - threshold;
        const double scale = 1.0 / tolerance;
        return 0.5 * (std::tanh(scale * diff) + 1.0);
    };

    const double cond1 = smoothCondition(dp_fan, p_max + tolerance);

    const double cond2 = smoothCondition(dp_fan, p1 + tolerance) * (1.0 - cond1);
    double flow2 = 0.0;
    const double p_diff = p1 - p_max;
    if (edgeData.p1 == edgeData.p_max) {
        flow2 = q1;
    } else {
        flow2 = q1 * (dp_fan - p_max) / p_diff;
    }

    const double cond3 = smoothCondition(dp_fan, tolerance) * (1.0 - smoothCondition(dp_fan, p1 + tolerance));
    double flow3 = 0.0;
    if (edgeData.p1 == 0.0) {
        flow3 = q_max;
    } else {
        flow3 = q1 + (q_max - q1) * (dp_fan - p1) / (-p1);
    }

    const double cond4 = 1.0 - smoothCondition(dp_fan, tolerance);

    return cond1 * 0.0 + cond2 * flow2 + cond3 * flow3 + cond4 * q_max;
}

// 統一風量計算インターフェース（ブランチタイプに応じて適切な計算関数を呼び出す）
inline double calculateUnifiedFlow(double dp, const EdgeProperties& edgeData) {
    if (edgeData.type == "simple_opening") {
        return calcSimpleOpeningFlow(dp, edgeData);
    } else if (edgeData.type == "gap") {
        return calcGapFlow(dp, edgeData);
    } else if (edgeData.type == "fan") {
        return calcFanFlow(dp, edgeData);
    } else if (edgeData.type == "fixed_flow") {
        return edgeData.current_vol;
    } else {
        return 0.0;
    }
}

} // namespace FlowCalculation


#pragma once

#include "../vtsim_solver.h"
#include "../../archenv/include/archenv.h"
#include <cmath>

// =============================================================================
// FlowCalculation - 風量計算の共通テンプレート関数群
// =============================================================================

namespace FlowCalculation {

// 開口部の風量計算（simple_opening）
template <typename T>
T calcSimpleOpeningFlow(const T& dp, const EdgeProperties& edgeData) {
    const T alpha = T(edgeData.alpha);
    const T area  = T(edgeData.area);
    const T eps   = T(archenv::TOLERANCE_SMALL);
    const T abs_dp = (dp >= T(0.0)) ? dp : -dp;
    const T K = alpha * area * sqrt(T(2.0) / T(archenv::DENSITY_DRY_AIR));
    if (abs_dp >= eps) {
    T sign = (dp >= T(0.0)) ? T(1.0) : T(-1.0);
        return sign * K * sqrt(abs_dp);
    }
    return K * sqrt(eps) * (dp / eps);
}

// 隙間の風量計算（gap）
template <typename T>
T calcGapFlow(const T& dp, const EdgeProperties& edgeData) {
    const T a  = T(edgeData.a);
    T n  = T(edgeData.n);
    if (n == T(0.0)) n = T(1.0);
    const T eps = T(archenv::TOLERANCE_SMALL);
    const T abs_dp = (dp >= T(0.0)) ? dp : -dp;
    if (abs_dp >= eps) {
    T sign = (dp >= T(0.0)) ? T(1.0) : T(-1.0);
        return sign * a * pow(abs_dp, T(1.0) / n);
    }
    return a * pow(eps, T(1.0) / n - T(1.0)) * dp;
}

// ファンの風量計算（fan）
template <typename T>
T calcFanFlow(const T& dp, const EdgeProperties& edgeData) {
    T q_max = T(edgeData.q_max);
    T p_max = T(edgeData.p_max);
    T p1 = T(edgeData.p1);
    T q1 = T(edgeData.q1);

    T dp_fan = -dp;
    T tolerance = T(archenv::TOLERANCE_MEDIUM);

    auto smoothCondition = [tolerance](const T& x, const T& threshold) -> T {
        T diff = x - threshold;
        T scale = T(1.0) / tolerance;
        return T(0.5) * (tanh(scale * diff) + T(1.0));
    };

    T cond1 = smoothCondition(dp_fan, p_max + tolerance);

    T cond2 = smoothCondition(dp_fan, p1 + tolerance) * (T(1.0) - cond1);
    T flow2;
    T p_diff = p1 - p_max;
    if (edgeData.p1 == edgeData.p_max) {
        flow2 = q1;
    } else {
        flow2 = q1 * (dp_fan - p_max) / p_diff;
    }

    T cond3 = smoothCondition(dp_fan, tolerance) * (T(1.0) - smoothCondition(dp_fan, p1 + tolerance));
    T flow3;
    if (edgeData.p1 == 0.0) {
        flow3 = q_max;
    } else {
        flow3 = q1 + (q_max - q1) * (dp_fan - p1) / (-p1);
    }

    T cond4 = T(1.0) - smoothCondition(dp_fan, tolerance);

    return cond1 * T(0.0) + cond2 * flow2 + cond3 * flow3 + cond4 * q_max;
}

// 統一風量計算インターフェース（ブランチタイプに応じて適切な計算関数を呼び出す）
template <typename T>
T calculateUnifiedFlow(const T& dp, const EdgeProperties& edgeData) {
    if (edgeData.type == "simple_opening") {
        return calcSimpleOpeningFlow(dp, edgeData);
    } else if (edgeData.type == "gap") {
        return calcGapFlow(dp, edgeData);
    } else if (edgeData.type == "fan") {
        return calcFanFlow(dp, edgeData);
    } else if (edgeData.type == "fixed_flow") {
        return T(edgeData.current_vol);
    } else {
        return T(0.0);
    }
}

} // namespace FlowCalculation


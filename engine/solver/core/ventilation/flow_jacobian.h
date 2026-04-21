#pragma once

#include "../../vtsim_solver.h"
#include "core/ventilation/flow_calculation.h"
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

// pressure_loss のヤコビアン（解析）
inline double calcPressureLossJacobian(double dp, const EdgeProperties& edgeData) {
    const double eps = archenv::TOLERANCE_SMALL;
    const double area = edgeData.area;

    double k_total = edgeData.k_total;
    if (!(k_total > 0.0) && edgeData.friction_factor > 0.0 && edgeData.length >= 0.0 && edgeData.diameter > 0.0) {
        k_total = edgeData.friction_factor * edgeData.length / edgeData.diameter + edgeData.zeta_total;
    }
    if (!(area > 0.0) || !(k_total > 0.0)) {
        return 0.0;
    }

    const double abs_dp = std::abs(dp);
    const double C = area * std::sqrt(2.0 / (archenv::DENSITY_DRY_AIR * k_total));
    if (abs_dp >= eps) {
        return 0.5 * C / std::sqrt(abs_dp);
    }
    return C * std::sqrt(eps) / eps;
}
} // namespace FlowJacobian

namespace FanJacobian {
// fanのヤコビアン（FlowCalculation::calcFanFlow のスムージング(tanh)に一致させる）
inline double calcFanJacobian(double dp, const EdgeProperties& edgeData) {
    const double q_max = edgeData.q_max;
    const double p_max = edgeData.p_max;
    const double p1 = edgeData.p1;
    const double q1 = edgeData.q1;

    const double x = -dp; // dp_fan
    const double tol = archenv::TOLERANCE_MEDIUM;
    if (!(tol > 0.0)) return 0.0;

    auto smooth = [tol](double x0, double thr) {
        const double t = (x0 - thr) / tol;
        return 0.5 * (std::tanh(t) + 1.0);
    };
    auto dsmooth_dx = [tol](double x0, double thr) {
        const double t = (x0 - thr) / tol;
        const double th = std::tanh(t);
        // sech^2(t) = 1 - tanh^2(t)
        return 0.5 * (1.0 - th * th) * (1.0 / tol);
    };

    const double s1  = smooth(x, p_max + tol);
    const double ds1 = dsmooth_dx(x, p_max + tol);

    const double s2  = smooth(x, p1 + tol);
    const double ds2 = dsmooth_dx(x, p1 + tol);

    const double s0  = smooth(x, tol);
    const double ds0 = dsmooth_dx(x, tol);

    const double cond1 = s1;
    const double cond2 = s2 * (1.0 - cond1);
    const double cond3 = s0 * (1.0 - s2);
    const double dcond1 = ds1;
    const double dcond2 = ds2 * (1.0 - cond1) + s2 * (-dcond1);
    const double dcond3 = ds0 * (1.0 - s2) + s0 * (-ds2);
    const double dcond4 = -ds0;

    double flow2 = 0.0;
    double dflow2_dx = 0.0;
    if (edgeData.p1 == edgeData.p_max) {
        flow2 = q1;
        dflow2_dx = 0.0;
    } else {
        const double denom = (p1 - p_max);
        flow2 = q1 * (x - p_max) / denom;
        dflow2_dx = q1 / denom;
    }

    double flow3 = 0.0;
    double dflow3_dx = 0.0;
    if (edgeData.p1 == 0.0) {
        flow3 = q_max;
        dflow3_dx = 0.0;
    } else {
        flow3 = q1 + (q_max - q1) * (x - p1) / (-p1);
        dflow3_dx = (q_max - q1) / (-p1);
    }

    // flow = cond2*flow2 + cond3*flow3 + cond4*q_max
    const double dflow_dx =
        dcond2 * flow2 + cond2 * dflow2_dx +
        dcond3 * flow3 + cond3 * dflow3_dx +
        dcond4 * q_max;

    const double dflow_ddp = -dflow_dx;
    if (!std::isfinite(dflow_ddp)) return 0.0;
    return dflow_ddp;
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
    } else if (edgeData.type == "pressure_loss") {
        return FlowJacobian::calcPressureLossJacobian(dp, edgeData);
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



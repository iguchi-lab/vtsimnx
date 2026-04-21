#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/flow_jacobian.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) +
             ", diff=" + std::to_string(diff) + ", tol=" + std::to_string(tol) + ")");
    }
}

} // namespace

int main() {
    // -----------------------------
    // simple_opening: 偶奇性（符号）と連続性（eps境界）とヤコビアン整合
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "simple_opening";
        e.alpha = 0.6;
        e.area = 1.2;

        const double eps = archenv::TOLERANCE_SMALL;
        const double K = e.alpha * e.area * std::sqrt(2.0 / archenv::DENSITY_DRY_AIR);

        // odd symmetry
        const double dp = 12.3;
        const double q_pos = FlowCalculation::calculateUnifiedFlow(dp, e);
        const double q_neg = FlowCalculation::calculateUnifiedFlow(-dp, e);
        expectNear(q_neg, -q_pos, 1e-12, "simple_opening: Q(-dp) == -Q(dp)");

        // continuity at abs_dp == eps
        const double q_linear = K * std::sqrt(eps) * ((0.5 * eps) / eps);
        const double q_small = FlowCalculation::calculateUnifiedFlow(0.5 * eps, e);
        expectNear(q_small, q_linear, 1e-12, "simple_opening: small-dp linearization");

        // Jacobian matches numerical derivative (excluding fan)
        auto f = [&](double x) { return FlowCalculation::calculateUnifiedFlow(x, e); };
        const double h = 1e-8;
        const double d_num = (f(dp + h) - f(dp - h)) / (2.0 * h);
        const double d_ana = FlowJacobianCommon::calculateJacobian(dp, e);
        expectTrue(std::isfinite(d_ana), "simple_opening: Jacobian is finite");
        expectNear(d_ana, d_num, 1e-6, "simple_opening: Jacobian matches numeric derivative");
    }

    // -----------------------------
    // gap: odd symmetry + n=1 special + Jacobian
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "gap";
        e.a = 0.02;
        e.n = 1.0;

        const double dp = 5.0;
        const double q_pos = FlowCalculation::calculateUnifiedFlow(dp, e);
        const double q_neg = FlowCalculation::calculateUnifiedFlow(-dp, e);
        expectNear(q_neg, -q_pos, 1e-12, "gap: Q(-dp) == -Q(dp)");

        // n=1 => Q = a*dp (outside eps region too)
        expectNear(q_pos, e.a * dp, 1e-9, "gap: n=1 gives linear flow");

        auto f = [&](double x) { return FlowCalculation::calculateUnifiedFlow(x, e); };
        const double h = 1e-8;
        const double d_num = (f(dp + h) - f(dp - h)) / (2.0 * h);
        const double d_ana = FlowJacobianCommon::calculateJacobian(dp, e);
        expectTrue(std::isfinite(d_ana), "gap: Jacobian is finite");
        expectNear(d_ana, d_num, 1e-6, "gap: Jacobian matches numeric derivative");
    }

    // -----------------------------
    // fixed_flow: Jacobian is 0
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "fixed_flow";
        e.current_vol = 1.23;

        const double dp = 10.0;
        expectNear(FlowCalculation::calculateUnifiedFlow(dp, e), e.current_vol, 0.0, "fixed_flow: Q is constant");
        expectNear(FlowJacobianCommon::calculateJacobian(dp, e), 0.0, 0.0, "fixed_flow: dQ/dp == 0");
    }

    // -----------------------------
    // pressure_loss: sqrt(dp) + Jacobian
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "pressure_loss";
        e.area = 0.1;
        e.k_total = 4.0;

        const double dp = 50.0;
        const double q_pos = FlowCalculation::calculateUnifiedFlow(dp, e);
        const double q_neg = FlowCalculation::calculateUnifiedFlow(-dp, e);
        expectNear(q_neg, -q_pos, 1e-12, "pressure_loss: Q(-dp) == -Q(dp)");

        auto f = [&](double x) { return FlowCalculation::calculateUnifiedFlow(x, e); };
        const double h = 1e-8;
        const double d_num = (f(dp + h) - f(dp - h)) / (2.0 * h);
        const double d_ana = FlowJacobianCommon::calculateJacobian(dp, e);
        expectTrue(std::isfinite(d_ana), "pressure_loss: Jacobian is finite");
        expectNear(d_ana, d_num, 1e-6, "pressure_loss: Jacobian matches numeric derivative");
    }

    // -----------------------------
    // fan: 解析ヤコビアンが数値微分と一致（スムージング込み）
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "fan";
        e.q_max = 1.0;
        e.p_max = 200.0;
        e.p1 = 100.0;
        e.q1 = 0.4;

        auto f = [&](double dp) { return FlowCalculation::calculateUnifiedFlow(dp, e); };
        auto dnum = [&](double dp) {
            // 中心差分（hはスムージング幅(tol)に比例させて、しきい値付近で過剰に跨がないようにする）
            const double tol = archenv::TOLERANCE_MEDIUM;
            double h = 1e-6;
            if (tol > 0.0) {
                h = std::max(1e-8, tol * 1e-3);
                h = std::min(h, tol * 0.05);
            }
            return (f(dp + h) - f(dp - h)) / (2.0 * h);
        };

        const double tol = archenv::TOLERANCE_MEDIUM;
        const double pmax = e.p_max;
        const double p1 = e.p1;

        // dp_fan = -dp のしきい値近傍も含めて複数点で確認
        const double testPoints[] = {
            -(pmax + (tol > 0.0 ? 2.0 * tol : 1.0)), // dp_fan > p_max 近傍（Q≈0）
            -(p1 + (tol > 0.0 ? 2.0 * tol : 1.0)),   // p1 近傍
            -((tol > 0.0 ? 2.0 * tol : 1.0)),        // 0 近傍（Q≈q_max）
            +50.0,                                    // dp_fan < 0（Q≈q_max）
            -150.0                                    // 中間域
        };

        for (double dp : testPoints) {
            const double d_ana = FlowJacobianCommon::calculateJacobian(dp, e);
            const double d_num = dnum(dp);
            expectTrue(std::isfinite(d_ana), "fan: Jacobian is finite");
            expectTrue(std::isfinite(d_num), "fan: numeric derivative is finite");
            // しきい値近傍はカーブが強く、数値微分誤差が出やすいのでやや広めに許容
            expectNear(d_ana, d_num, 1e-3, "fan: Jacobian matches numeric derivative");
        }
    }

    // -----------------------------
    // unknown type: fallback numerical derivative should return 0 (Q is 0)
    // -----------------------------
    {
        EdgeProperties e{};
        e.type = "unknown_type";
        const double dp = 10.0;
        expectNear(FlowCalculation::calculateUnifiedFlow(dp, e), 0.0, 0.0, "unknown: Q == 0");
        expectNear(FlowJacobianCommon::calculateJacobian(dp, e), 0.0, 0.0, "unknown: fallback Jacobian == 0");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}



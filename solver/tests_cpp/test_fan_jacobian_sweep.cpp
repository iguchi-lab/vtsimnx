#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
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

double numericDerivative(const EdgeProperties& e, double dp) {
    const double tol = archenv::TOLERANCE_MEDIUM;
    double h = 1e-6;
    if (tol > 0.0) {
        // スムージング幅の 0.1% 〜 5% に制限（しきい値近傍で跨ぎすぎない）
        h = std::max(1e-8, tol * 1e-3);
        h = std::min(h, tol * 0.05);
    }
    auto f = [&](double x) { return FlowCalculation::calculateUnifiedFlow(x, e); };
    return (f(dp + h) - f(dp - h)) / (2.0 * h);
}

} // namespace

int main() {
    std::mt19937 rng(123456u);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    auto randIn = [&](double lo, double hi) {
        return lo + (hi - lo) * uni01(rng);
    };

    const double tol = archenv::TOLERANCE_MEDIUM;
    const int cases = 50;
    const int pointsPerCase = 12;

    for (int c = 0; c < cases; ++c) {
        EdgeProperties e{};
        e.type = "fan";

        e.p_max = randIn(50.0, 500.0);
        e.p1 = randIn(1.0, std::max(2.0, e.p_max - 1.0));
        if (e.p1 >= e.p_max) e.p1 = std::max(1.0, e.p_max * 0.5);

        e.q_max = randIn(0.2, 5.0);
        e.q1 = randIn(0.0, e.q_max);

        // しきい値近傍 + 広い範囲を混ぜて評価
        const double thr[] = {e.p_max, e.p1, 0.0};
        for (int i = 0; i < pointsPerCase; ++i) {
            double dp_fan = 0.0;
            if (i < 6) {
                const double t = thr[i % 3];
                const double w = (tol > 0.0) ? (3.0 * tol) : 1.0;
                dp_fan = t + randIn(-w, +w);
            } else {
                dp_fan = randIn(-1.5 * e.p_max, 1.5 * e.p_max);
            }
            const double dp = -dp_fan;

            const double d_ana = FlowJacobianCommon::calculateJacobian(dp, e);
            const double d_num = numericDerivative(e, dp);

            expectTrue(std::isfinite(d_ana), "fan sweep: analytic jacobian finite");
            expectTrue(std::isfinite(d_num), "fan sweep: numeric derivative finite");

            // しきい値付近は曲率が高いので少し広め
            expectNear(d_ana, d_num, 2e-3, "fan sweep: analytic matches numeric");
        }
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}



#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "acmodel/refrigerant_calculator.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectFinite(double v, const std::string& msg) {
    if (!std::isfinite(v)) {
        fail(msg + " (non-finite)");
    }
}

void expectNear(double actual, double expected, double tol, const std::string& msg) {
    const double diff = std::abs(actual - expected);
    if (diff > tol) {
        fail(msg + " (actual=" + std::to_string(actual) +
             ", expected=" + std::to_string(expected) +
             ", tol=" + std::to_string(tol) + ")");
    }
}

} // namespace

int main() {
    using acmodel::RefrigerantCalculator;

    // -----------------------------
    // 1) 飽和圧力: 正値・有限・温度単調増加
    // -----------------------------
    {
        const std::vector<double> temps = {-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0};
        double prev = RefrigerantCalculator::getSaturatedGasPressure(temps.front());
        expectFinite(prev, "saturated pressure finite at first point");
        expectTrue(prev > 0.0, "saturated pressure positive at first point");

        for (size_t i = 1; i < temps.size(); ++i) {
            const double now = RefrigerantCalculator::getSaturatedGasPressure(temps[i]);
            expectFinite(now, "saturated pressure finite");
            expectTrue(now > 0.0, "saturated pressure positive");
            expectTrue(now > prev, "saturated pressure increases with temperature");
            prev = now;
        }
    }

    // -----------------------------
    // 2) 圧縮機吸込エンタルピー: 固定圧で温度単調増加
    // -----------------------------
    {
        const double P = 1.6;
        const std::vector<double> temps = {-20.0, -10.0, 0.0, 10.0, 20.0, 30.0};
        double prev = RefrigerantCalculator::getGasCompressorInletEnthalpy(P, temps.front());
        expectFinite(prev, "inlet enthalpy finite at first point");

        for (size_t i = 1; i < temps.size(); ++i) {
            const double now = RefrigerantCalculator::getGasCompressorInletEnthalpy(P, temps[i]);
            expectFinite(now, "inlet enthalpy finite");
            expectTrue(now > prev, "inlet enthalpy increases with temperature");
            prev = now;
        }
    }

    // -----------------------------
    // 3) 液エンタルピー: 固定圧で温度単調増加
    // -----------------------------
    {
        const double P = 2.4;
        const std::vector<double> temps = {-10.0, 0.0, 10.0, 20.0, 30.0, 40.0};
        double prev = RefrigerantCalculator::getLiquidEnthalpy(P, temps.front());
        expectFinite(prev, "liquid enthalpy finite at first point");

        for (size_t i = 1; i < temps.size(); ++i) {
            const double now = RefrigerantCalculator::getLiquidEnthalpy(P, temps[i]);
            expectFinite(now, "liquid enthalpy finite");
            expectTrue(now > prev, "liquid enthalpy increases with temperature");
            prev = now;
        }
    }

    // -----------------------------
    // 4) 圧縮機吐出エンタルピー: 固定圧でエントロピー単調増加
    // -----------------------------
    {
        const double P = 2.5;
        const std::vector<double> entropies = {1.2, 1.4, 1.6, 1.8, 2.0, 2.2};
        double prev = RefrigerantCalculator::getGasCompressorOutletEnthalpy(P, entropies.front());
        expectFinite(prev, "outlet enthalpy finite at first point");

        for (size_t i = 1; i < entropies.size(); ++i) {
            const double now = RefrigerantCalculator::getGasCompressorOutletEnthalpy(P, entropies[i]);
            expectFinite(now, "outlet enthalpy finite");
            expectTrue(now > prev, "outlet enthalpy increases with entropy");
            prev = now;
        }
    }

    // -----------------------------
    // 5) 圧縮仕事の符号: P_out > P_in なら h_out > h_in
    // -----------------------------
    {
        const double thetaEvp = 7.0;
        const double thetaCnd = 35.0;
        const double thetaSH = 5.0;

        const double pIn = RefrigerantCalculator::getSaturatedGasPressure(thetaEvp);
        const double pOut = RefrigerantCalculator::getSaturatedGasPressure(thetaCnd);
        const double hIn = RefrigerantCalculator::getGasCompressorInletEnthalpy(pIn, thetaEvp + thetaSH);
        const double sIn = RefrigerantCalculator::getGasEntropy(pIn, hIn);
        const double hOut = RefrigerantCalculator::getGasCompressorOutletEnthalpy(pOut, sIn);

        expectFinite(hIn, "compressor h_in finite");
        expectFinite(hOut, "compressor h_out finite");
        expectTrue(pOut > pIn, "compressor test condition p_out > p_in");
        expectTrue(hOut > hIn, "compressor discharge enthalpy should exceed inlet enthalpy");
    }

    // -----------------------------
    // 6) 理論暖房効率: 温度リフト増大で低下（傾向）
    // -----------------------------
    {
        const double lowLift = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
            10.0, 35.0, 4.0, 4.0);
        const double highLift = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
            -5.0, 50.0, 4.0, 4.0);

        expectFinite(lowLift, "heating efficiency low-lift finite");
        expectFinite(highLift, "heating efficiency high-lift finite");
        expectTrue(lowLift > 0.0, "heating efficiency low-lift positive");
        expectTrue(highLift > 0.0, "heating efficiency high-lift positive");
        expectTrue(lowLift > highLift, "heating efficiency should decrease as temperature lift increases");
    }

    // -----------------------------
    // 7) 有効レンジ内スイープ: 効率は有限かつ正
    // -----------------------------
    {
        const std::vector<double> evps = {-15.0, -5.0, 0.0, 5.0, 10.0};
        const std::vector<double> cnds = {30.0, 40.0, 50.0, 60.0};

        for (double evp : evps) {
            for (double cnd : cnds) {
                if (cnd <= evp + 5.0) continue;
                const double e = RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(
                    evp, cnd, 4.0, 4.0);
                expectFinite(e, "sweep: heating efficiency finite");
                expectTrue(e > 0.0, "sweep: heating efficiency positive");
            }
        }
    }

    // -----------------------------
    // 8) 固定値回帰: 代表点の数値を保持（係数改変の破壊検知）
    // -----------------------------
    {
        // 多項式評価は丸め影響が小さいため、比較的厳しめの許容値にする。
        expectNear(RefrigerantCalculator::getSaturatedGasPressure(-20.0), 0.3993007948514956, 1.0e-12,
                   "regression: p_sat(-20C)");
        expectNear(RefrigerantCalculator::getSaturatedGasPressure(0.0), 0.7980864551547750, 1.0e-12,
                   "regression: p_sat(0C)");
        expectNear(RefrigerantCalculator::getSaturatedGasPressure(20.0), 1.4429236674606490, 1.0e-12,
                   "regression: p_sat(20C)");
        expectNear(RefrigerantCalculator::getSaturatedGasPressure(40.0), 2.4186086248145190, 1.0e-12,
                   "regression: p_sat(40C)");

        expectNear(RefrigerantCalculator::getGasCompressorInletEnthalpy(1.6, 15.0), 416.23200849420454, 1.0e-9,
                   "regression: h_in(P=1.6, T=15C)");
        expectNear(RefrigerantCalculator::getGasCompressorOutletEnthalpy(2.5, 1.8), 449.06388070651155, 1.0e-9,
                   "regression: h_out(P=2.5, S=1.8)");
        expectNear(RefrigerantCalculator::getGasEntropy(1.4, 380.0), 1.6277163880185697, 1.0e-9,
                   "regression: s(P=1.4, h=380)");
        expectNear(RefrigerantCalculator::getLiquidEnthalpy(2.2, 25.0), 239.69627082538150, 1.0e-9,
                   "regression: h_liq(P=2.2, T=25C)");

        expectNear(RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(7.0, 35.0, 4.0, 4.0),
                   9.405956474940092, 1.0e-9,
                   "regression: e_th(Tevp=7, Tcnd=35, SC=4, SH=4)");
        expectNear(RefrigerantCalculator::calculateTheoreticalHeatingEfficiency(-5.0, 45.0, 4.0, 4.0),
                   4.959087551974653, 1.0e-9,
                   "regression: e_th(Tevp=-5, Tcnd=45, SC=4, SH=4)");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


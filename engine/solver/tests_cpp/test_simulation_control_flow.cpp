#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "parser/sim_constants_parser.h"
#include "simulation_aircon_iteration.h"
#include "simulation_coupling_control.h"
#include "simulation_error.h"
#include "simulation_inner_coupling.h"
#include "simulation_runner_helpers.h"
#include "types/common_types.h"

namespace {

using simulation::AirconIterationAction;
using simulation::decideAirconIterationAction;
using simulation::resolveLatentAppliedThisIter;
using simulation::LatentCouplingMode;
using simulation::detail::CoupledDelta;
using simulation::detail::InnerCouplingAction;
using simulation::detail::InnerCouplingEval;
using simulation::detail::effectiveMaxAirconControlIterations;
using simulation::detail::effectiveMaxCouplingIterations;
using simulation::detail::ensureOuterAirconLoopConverged;
using simulation::detail::evaluateInnerCoupling;

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
    if (diff > tol) {
        fail(msg + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

void expectEqAction(InnerCouplingAction actual, InnerCouplingAction expected, const std::string& msg) {
    if (actual != expected) {
        fail(msg + " (action mismatch)");
    }
}

void expectEqAircon(AirconIterationAction actual, AirconIterationAction expected, const std::string& msg) {
    if (actual != expected) {
        fail(msg + " (aircon action mismatch)");
    }
}

SimulationConstants baseConstants() {
    SimulationConstants c{};
    c.maxInnerIterations = 5;
    c.maxCouplingIterations = 5;
    c.maxAirconControlIterations = 3;
    c.pressureCalc = true;
    c.temperatureCalc = true;
    c.humidityCalc = false;
    c.moistureCouplingEnabled = true;
    c.convergenceTolerance = 1e-3;
    c.couplingPressureTolerance = 1e-3;
    c.couplingTemperatureTolerance = 1e-3;
    c.couplingHumidityTolerance = 1e-3;
    return c;
}

CoupledDelta largeDelta() {
    CoupledDelta d{};
    d.pressureChange = 1.0;
    d.temperatureChange = 1.0;
    d.humidityChange = 1.0;
    return d;
}

CoupledDelta tinyDelta() {
    CoupledDelta d{};
    d.pressureChange = 1e-9;
    d.temperatureChange = 1e-9;
    d.humidityChange = 1e-9;
    return d;
}

void testInnerCouplingBreakNoNeed() {
    auto c = baseConstants();
    c.pressureCalc = true;
    c.temperatureCalc = false;
    c.humidityCalc = false;
    const auto eval = evaluateInnerCoupling(c, false, 1, largeDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::BreakNoNeed, "inner: no-need exits after first iter");
}

void testInnerCouplingConvergedOnSecond() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, 2, tinyDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::BreakConverged, "inner: converge on second iter");
}

void testInnerCouplingContinueOnFirst() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, 1, largeDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::Continue, "inner: first coupled iter continues");
}

void testInnerCouplingPressureFirstFail() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, 1, largeDelta(), false);
    expectEqAction(eval.action, InnerCouplingAction::ThrowPressureNonConvergence,
                   "inner: pressure non-convergence on first iter");
}

void testInnerCouplingMaxIterations() {
    auto c = baseConstants();
    c.maxCouplingIterations = 2;
    const auto eval = evaluateInnerCoupling(c, false, 2, largeDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::ThrowMaxIteration, "inner: max coupling iterations");
}

void testHumidityCouplingFlags() {
    auto c = baseConstants();
    c.humidityCalc = true;
    c.moistureCouplingEnabled = true;
    expectTrue(simulation::detail::humidityCouplingActive(c), "humidity coupling ON");
    c.moistureCouplingEnabled = false;
    expectTrue(!simulation::detail::humidityCouplingActive(c), "humidity coupling OFF");
}

void testAirconDecideActions() {
    expectEqAircon(decideAirconIterationAction(true, true, true), AirconIterationAction::RecomputeForFlow,
                   "aircon: flow wins");
    expectEqAircon(decideAirconIterationAction(false, false, true), AirconIterationAction::RecomputeForControl,
                   "aircon: control wins");
    expectEqAircon(decideAirconIterationAction(false, true, true), AirconIterationAction::RecomputeForCapacity,
                   "aircon: capacity");
    expectEqAircon(decideAirconIterationAction(false, true, false), AirconIterationAction::Accept,
                   "aircon: accept");
}

void testOuterMaxIterationsThrow() {
    const std::size_t maxOuter = effectiveMaxAirconControlIterations(baseConstants());
    bool outerLoopConverged = false;
    for (std::size_t iteration = 0; iteration < maxOuter; ++iteration) {
        const auto action = AirconIterationAction::RecomputeForCapacity;
        if (action != AirconIterationAction::Accept) {
            continue;
        }
        outerLoopConverged = true;
        break;
    }

    bool threw = false;
    try {
        ensureOuterAirconLoopConverged(outerLoopConverged);
    } catch (const simulation::Error& e) {
        threw = true;
        expectTrue(e.code() == simulation::ErrorCode::AirconMaxIterations, "outer: error code");
        expectTrue(std::string(simulation::toErrorCodeString(e.code())) == "aircon_max_iterations",
                   "outer: error_code string");
        expectTrue(std::string(e.what()).find("Aircon control did not converge") != std::string::npos,
                   "outer: message");
    }
    expectTrue(threw, "outer: must throw AirconMaxIterations");
    expectTrue(!outerLoopConverged, "outer: remained unconverged");
}

void testOuterAcceptDoesNotThrow() {
    bool threw = false;
    try {
        ensureOuterAirconLoopConverged(true);
    } catch (...) {
        threw = true;
    }
    expectTrue(!threw, "outer: Accept path must not throw");
}

void testLatentModeResolve() {
    expectTrue(resolveLatentAppliedThisIter(LatentCouplingMode::Disabled) == 0.0,
               "latent: Disabled returns 0");
    bool threw = false;
    try {
        (void)resolveLatentAppliedThisIter(LatentCouplingMode::FeedbackToThermal);
    } catch (const std::logic_error&) {
        threw = true;
    }
    expectTrue(threw, "latent: FeedbackToThermal must throw logic_error");
}

void testEffectiveIterationFallback() {
    SimulationConstants c{};
    c.maxInnerIterations = 7;
    c.maxCouplingIterations = 0;
    c.maxAirconControlIterations = 0;
    expectTrue(effectiveMaxCouplingIterations(c) == 7, "fallback coupling");
    expectTrue(effectiveMaxAirconControlIterations(c) == 7, "fallback aircon");
}

nlohmann::json minimalSimJson() {
    return nlohmann::json{
        {"simulation",
         {{"index",
           {{"start", "0"}, {"end", "1"}, {"timestep", 1}, {"length", 1}}},
          {"tolerance",
           {{"ventilation", 1e-6}, {"thermal", 1e-6}, {"convergence", 1e-6}}},
          {"calc_flag", {{"p", true}, {"t", true}, {"x", false}, {"c", false}}}}},
    };
}

void testParserPositiveIntegerIterations() {
    std::ostringstream logs;

    {
        auto j = minimalSimJson();
        j["simulation"]["iteration"] = {{"max_inner", 12}};
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.maxInnerIterations == 12, "parser: max_inner integer");
        expectTrue(c.maxCouplingIterations == 12, "parser: coupling default copy");
        expectTrue(c.maxAirconControlIterations == 12, "parser: aircon default copy");
    }

    {
        auto j = minimalSimJson();
        j["simulation"]["tolerance"]["thermal"] = 1e-3;
        j["simulation"]["tolerance"]["aircon_temperature"] = 0.5;
        j["simulation"]["tolerance"]["thermal_balance"] = 2.0;
        j["simulation"]["tolerance"]["thermal_linear_residual"] = 1e-8;
        const auto c = parseSimulationConstants(j, logs);
        expectNear(c.thermalTolerance, 1e-3, 0.0, "parser: thermal compat");
        expectNear(c.airconTemperatureToleranceK, 0.5, 0.0, "parser: aircon_temperature");
        expectNear(c.thermalBalanceToleranceW, 2.0, 0.0, "parser: thermal_balance");
        expectNear(c.thermalLinearResidualRelativeTolerance, 1e-8, 0.0,
                   "parser: thermal_linear_residual");
        expectNear(effectiveAirconTemperatureToleranceK(c), 0.5, 0.0, "effective aircon");
        expectNear(effectiveThermalBalanceToleranceW(c), 2.0, 0.0, "effective balance");
        expectNear(effectiveThermalLinearResidualRelativeTolerance(c), 1e-8, 0.0,
                   "effective linear residual");
    }

    {
        auto j = minimalSimJson();
        j["simulation"]["iteration"] = {{"max_inner", 3.8}};
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("positive integer") != std::string::npos,
                       "parser: float rejected message");
        }
        expectTrue(threw, "parser: reject non-integer max_inner");
    }

    {
        auto j = minimalSimJson();
        j["simulation"]["iteration"] = {{"max_inner", 0}};
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expectTrue(threw, "parser: reject non-positive max_inner");
    }
}

} // namespace

int main() {
    testInnerCouplingBreakNoNeed();
    testInnerCouplingConvergedOnSecond();
    testInnerCouplingContinueOnFirst();
    testInnerCouplingPressureFirstFail();
    testInnerCouplingMaxIterations();
    testHumidityCouplingFlags();
    testAirconDecideActions();
    testOuterMaxIterationsThrow();
    testOuterAcceptDoesNotThrow();
    testLatentModeResolve();
    testEffectiveIterationFallback();
    testParserPositiveIntegerIterations();

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[DONE] failures=" << g_failures << "\n";
    return 1;
}

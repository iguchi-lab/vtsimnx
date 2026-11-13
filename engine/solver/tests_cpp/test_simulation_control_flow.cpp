#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
#include "simulation_metrics.h"
#include "simulation_runner_helpers.h"
#include "types/aircon_control_state.h"
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
    const auto eval = evaluateInnerCoupling(c, false, false, 1, 2, largeDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::BreakNoNeed, "inner: no-need exits after first iter");
}

void testInnerCouplingConvergedOnSecond() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, false, 2, 2, tinyDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::BreakConverged, "inner: converge on second iter");
}

void testInnerCouplingContinueOnFirst() {
    auto c = baseConstants();
    // min=2 のとき1回目は Continue
    const auto eval = evaluateInnerCoupling(c, false, false, 1, 2, largeDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::Continue, "inner: first coupled iter continues");
}

void testInnerCouplingAllowFirstWhenMinOne() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, false, 1, 1, tinyDelta(), true);
    expectEqAction(eval.action, InnerCouplingAction::BreakConverged, "inner: min=1 allows first-iter converge");
}

void testInnerCouplingPressureFirstFail() {
    auto c = baseConstants();
    const auto eval = evaluateInnerCoupling(c, false, false, 1, 2, largeDelta(), false);
    expectEqAction(eval.action, InnerCouplingAction::ThrowPressureNonConvergence,
                   "inner: pressure non-convergence on first iter");
}

void testTemperatureRelaxationAlpha() {
    using simulation::detail::couplingTemperatureRelaxation;
    using simulation::detail::kCouplingTemperatureMaxMixedStepK;
    using simulation::detail::kCouplingTemperatureStallRelaxation;
    const double tol = 1e-3;
    expectNear(couplingTemperatureRelaxation(1.0, 1, 12.0, 0.0, tol), 1.0,
               1e-12, "T-relax: first iter below cap stays 1");
    expectNear(couplingTemperatureRelaxation(1.0, 2, 0.01, 12.0, tol), 1.0,
               1e-12, "T-relax: decreasing delta stays 1");
    expectNear(couplingTemperatureRelaxation(1.0, 2, 0.005, 0.0052, tol), 1.0,
               1e-12, "T-relax: near-tol decrease is not stalled");
    expectNear(couplingTemperatureRelaxation(1.0, 2, 5.0, 5.0, tol),
               kCouplingTemperatureStallRelaxation, 1e-12,
               "T-relax: period-2 below cap uses 0.5");
    expectNear(couplingTemperatureRelaxation(1.0, 2, 59.5, 59.48, tol),
               kCouplingTemperatureMaxMixedStepK / 59.5, 1e-12,
               "T-relax: period-2 above cap uses step limit");
    const double huge = 597.0;
    const double capped = couplingTemperatureRelaxation(1.0, 1, huge, 0.0, tol);
    expectNear(capped, kCouplingTemperatureMaxMixedStepK / huge, 1e-12,
               "T-relax: first-iter cap on huge delta");
    expectTrue(capped < 0.05, "T-relax: huge delta is strongly damped");
    expectNear(couplingTemperatureRelaxation(0.3, 2, 59.5, 59.5, tol), 0.3, 1e-12,
               "T-relax: configured ceiling is respected");
    // A sub-kelvin oscillation can still be hundreds of times above tolerance.
    // Solve x = 1.01 - 1.02*x, whose undamped Picard iteration diverges.
    double x = 0.1;
    double previousDefect = 0.0;
    bool converged = false;
    for (std::size_t iter = 1; iter <= 100; ++iter) {
        const double candidate = 1.01 - 1.02 * x;
        const double defect = std::abs(candidate - x);
        if (defect < tol) {
            converged = true;
            break;
        }
        const double alpha = couplingTemperatureRelaxation(1.0, iter, defect, previousDefect, tol);
        x += alpha * (candidate - x);
        previousDefect = defect;
    }
    expectTrue(converged, "T-relax: sub-kelvin Picard cycle converges within the iteration budget");
    expectNear(x, 0.5, tol, "T-relax: converged state satisfies the fixed-point equation");
}

void testUnphysicalTemperatureDetection() {
    using simulation::detail::anyUnphysicalCalcTemperature;
    using simulation::detail::couplingTemperatureUnphysical;
    using simulation::detail::invalidateAirconRequiredHeat;
    using simulation::detail::kCouplingTemperatureAbsMax;
    using simulation::detail::kCouplingTemperatureAbsMin;

    expectTrue(!couplingTemperatureUnphysical(kCouplingTemperatureAbsMin),
               "T-bound: min is physical");
    expectTrue(!couplingTemperatureUnphysical(kCouplingTemperatureAbsMax),
               "T-bound: max is physical");
    expectTrue(!couplingTemperatureUnphysical(20.0), "T-bound: room T is physical");
    expectTrue(couplingTemperatureUnphysical(kCouplingTemperatureAbsMin - 0.1),
               "T-bound: below min is unphysical");
    expectTrue(couplingTemperatureUnphysical(kCouplingTemperatureAbsMax + 0.1),
               "T-bound: above max is unphysical");
    expectTrue(couplingTemperatureUnphysical(3479.0), "T-bound: 3479C is unphysical");
    expectTrue(couplingTemperatureUnphysical(std::numeric_limits<double>::quiet_NaN()),
               "T-bound: NaN is unphysical");
    expectTrue(couplingTemperatureUnphysical(std::numeric_limits<double>::infinity()),
               "T-bound: Inf is unphysical");

    Graph g;
    const auto room = boost::add_vertex(g);
    g[room].key = "ROOM";
    g[room].type = "normal";
    g[room].calc_t = true;
    g[room].current_t = 18.8;
    const auto wall = boost::add_vertex(g);
    g[wall].key = "WALL";
    g[wall].type = "layer";
    g[wall].calc_t = false;
    g[wall].current_t = 500.0;
    expectTrue(!anyUnphysicalCalcTemperature(g),
               "unphysical T: calc_t=false outliers are ignored");

    g[room].current_t = 206.0;
    expectTrue(anyUnphysicalCalcTemperature(g),
               "unphysical T: calc_t node at 206C is rejected");

    const auto ac = boost::add_vertex(g);
    g[ac].key = "AC";
    g[ac].type = "aircon";
    g[ac].required_heat_w = 61198.1;
    g[room].required_heat_w = 123.0;
    invalidateAirconRequiredHeat(g);
    expectTrue(!std::isfinite(g[ac].required_heat_w),
               "unphysical T: Qreq is cleared on aircon nodes");
    expectNear(g[room].required_heat_w, 123.0, 0.0,
               "unphysical T: non-aircon Qreq is unchanged");
}

void testUnphysicalHumidityDetection() {
    using simulation::detail::anyUnphysicalCalcHumidity;
    using simulation::detail::couplingHumidityUnphysical;
    using simulation::detail::kCouplingHumidityAbsMax;
    using simulation::detail::kCouplingHumidityAbsMin;

    expectTrue(!couplingHumidityUnphysical(0.0), "x-bound: 0 is physical");
    expectTrue(!couplingHumidityUnphysical(0.01), "x-bound: room humidity is physical");
    expectTrue(!couplingHumidityUnphysical(kCouplingHumidityAbsMax),
               "x-bound: max is physical");
    expectTrue(!couplingHumidityUnphysical(-1e-9),
               "x-bound: tiny negative is numerical noise");
    expectTrue(couplingHumidityUnphysical(kCouplingHumidityAbsMin - 1e-9),
               "x-bound: below min is unphysical");
    expectTrue(couplingHumidityUnphysical(kCouplingHumidityAbsMax + 0.01),
               "x-bound: above max is unphysical");
    expectTrue(couplingHumidityUnphysical(-0.01), "x-bound: clearly negative is unphysical");
    expectTrue(couplingHumidityUnphysical(std::numeric_limits<double>::quiet_NaN()),
               "x-bound: NaN is unphysical");
    expectTrue(couplingHumidityUnphysical(std::numeric_limits<double>::infinity()),
               "x-bound: Inf is unphysical");

    Graph g;
    const auto room = boost::add_vertex(g);
    g[room].key = "ROOM";
    g[room].type = "normal";
    g[room].calc_x = true;
    g[room].current_x = 0.008;
    const auto supply = boost::add_vertex(g);
    g[supply].key = "AC";
    g[supply].type = "normal";
    g[supply].calc_x = false;
    g[supply].current_x = 2.0;
    expectTrue(!anyUnphysicalCalcHumidity(g),
               "unphysical x: calc_x=false outliers are ignored");

    g[supply].type = "aircon";
    g[supply].on = false;
    expectTrue(anyUnphysicalCalcHumidity(g),
               "unphysical x: OFF aircon is a solved mixing node even with calc_x=false");
    g[supply].current_x = 0.008;
    expectTrue(!anyUnphysicalCalcHumidity(g),
               "physical room and mixing-node humidities are accepted");

    g[room].current_x = 0.8;
    expectTrue(anyUnphysicalCalcHumidity(g),
               "unphysical x: calc_x node at 0.8 kg/kg is rejected");
}

void testInnerCouplingMaxIterations() {
    auto c = baseConstants();
    c.maxCouplingIterations = 2;
    const auto eval = evaluateInnerCoupling(c, false, false, 2, 2, largeDelta(), true);
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
    expectEqAircon(decideAirconIterationAction(true, true, true), AirconIterationAction::RecomputeForCapacity,
                   "aircon: capacity wins over flow");
    expectEqAircon(decideAirconIterationAction(false, false, true), AirconIterationAction::RecomputeForControl,
                   "aircon: control wins");
    expectEqAircon(decideAirconIterationAction(false, true, true), AirconIterationAction::RecomputeForCapacity,
                   "aircon: capacity");
    expectEqAircon(decideAirconIterationAction(true, true, false), AirconIterationAction::RecomputeForFlow,
                   "aircon: flow after capacity settled");
    expectEqAircon(decideAirconIterationAction(false, true, false, true),
                   AirconIterationAction::RecomputeForSupplyHumidity, "aircon: supply humidity");
    expectEqAircon(decideAirconIterationAction(false, true, false), AirconIterationAction::Accept,
                   "aircon: accept");
    expectEqAircon(decideAirconIterationAction(false, true, true, true),
                   AirconIterationAction::RecomputeForCapacity, "aircon: capacity before supply");

    using simulation::reasonsFromAirconFlags;
    using simulation::recordAirconRecomputeMetrics;
    const auto mixedReasons = reasonsFromAirconFlags(true, false, true, true);
    expectEqAircon(decideAirconIterationAction(mixedReasons), AirconIterationAction::RecomputeForControl,
                   "aircon reasons: onoff wins when present");
    expectTrue(hasReason(mixedReasons, AirconRecomputeReason::AirflowChanged), "mask has airflow");
    expectTrue(hasReason(mixedReasons, AirconRecomputeReason::OnOffChanged), "mask has onoff");
    expectTrue(hasReason(mixedReasons, AirconRecomputeReason::CapacitySetpointChanged), "mask has capacity");
    expectTrue(hasReason(mixedReasons, AirconRecomputeReason::SupplyHumidityChanged), "mask has humidity");

    simulation::TimestepSolveMetrics metrics;
    recordAirconRecomputeMetrics(&metrics, mixedReasons);
    expectTrue(metrics.airconOnOffRecalc == 1, "primary metric is onoff");
    expectTrue(metrics.airconFlowAdjustRecalc == 0, "non-primary flow not counted");
    expectTrue(metrics.airconCapacityRecalc == 0, "non-primary capacity not counted");
    expectTrue(metrics.airconRecomputeReasonsMask == static_cast<std::uint32_t>(mixedReasons),
               "reasons mask ORed");

    const auto capacityFlowReasons = reasonsFromAirconFlags(true, true, true, false);
    expectEqAircon(decideAirconIterationAction(capacityFlowReasons),
                   AirconIterationAction::RecomputeForCapacity,
                   "aircon reasons: capacity before flow");
    simulation::TimestepSolveMetrics capacityMetrics;
    recordAirconRecomputeMetrics(&capacityMetrics, capacityFlowReasons);
    expectTrue(capacityMetrics.airconCapacityRecalc == 1, "primary metric is capacity over flow");
    expectTrue(capacityMetrics.airconFlowAdjustRecalc == 0, "flow not primary when capacity set");

    AirconStateProposal p1;
    p1.reasons = AirconRecomputeReason::CapacitySetpointChanged;
    AirconStateProposal p2;
    p2.reasons = AirconRecomputeReason::SupplyHumidityChanged;
    const auto agg = aggregateProposalReasons({p1, p2});
    expectEqAircon(decideAirconIterationAction(agg), AirconIterationAction::RecomputeForCapacity,
                   "aggregate proposals prefer capacity");

    simulation::TimestepSolveMetrics step;
    step.outerIterations = 4;
    step.airconCapacityRecalc = 3;
    step.airconOnOffRecalc = 1;
    step.finalizeTimestepOuterStats();
    expectTrue(step.solveTimesteps == 1, "finalize counts timestep");
    expectTrue(step.outerIterationsMax == 4, "finalize tracks max");
    expectTrue(step.outerIterationsGe3 == 1, "finalize tracks ge3");
    const auto j = step.toJson();
    expectTrue(j.contains("aircon_capacity_recalc_share"), "json has capacity share");
    expectTrue(std::abs(j["aircon_capacity_recalc_share"].get<double>() - 0.75) < 1e-12,
               "capacity share 3/4");
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
    expectNear(resolveLatentAppliedThisIter(LatentCouplingMode::FromHumidityChange, 12.5), 12.5, 0.0,
               "latent: FromHumidityChange returns applied W");
    expectNear(resolveLatentAppliedThisIter(LatentCouplingMode::FromPhaseChange, 3.0), 3.0, 0.0,
               "latent: FromPhaseChange returns applied W");
}

void testLatentFromHumidityChangeKnownDx() {
    using simulation::detail::updateLatentFromHumidityChange;

    Graph g;
    const auto v = boost::add_vertex(g);
    g[v].key = "ROOM";
    g[v].v = 10.0;
    g[v].current_x = 0.012;
    const std::vector<double> xN{0.010};
    const double dt = 2.0;
    const double rho = PhysicalConstants::DENSITY_DRY_AIR;
    const double L = PhysicalConstants::LATENT_HEAT_VAPORIZATION;
    const double expected = -rho * 10.0 * L * (0.012 - 0.010) / dt;

    std::vector<double> latent{0.0};
    updateLatentFromHumidityChange(g, xN, dt, /*relaxation=*/1.0, latent);
    expectNear(latent[0], expected, 1e-9, "latent: known dx formula");

    // V<=0 → 0
    g[v].v = 0.0;
    latent[0] = 123.0;
    updateLatentFromHumidityChange(g, xN, dt, 1.0, latent);
    expectNear(latent[0], 0.0, 0.0, "latent: V<=0 yields 0");

    // relaxation: Q = (1-a)*prev + a*raw
    g[v].v = 10.0;
    latent[0] = 100.0;
    updateLatentFromHumidityChange(g, xN, dt, 0.5, latent);
    expectNear(latent[0], 0.5 * 100.0 + 0.5 * expected, 1e-9, "latent: relaxation mix");

    // dt<=0 → no-op
    latent[0] = 42.0;
    updateLatentFromHumidityChange(g, xN, 0.0, 1.0, latent);
    expectNear(latent[0], 42.0, 0.0, "latent: dt<=0 no-op");
}

void testLatentCouplingActiveRequiresHumidity() {
    SimulationConstants c{};
    c.humidityCalc = true;
    c.temperatureCalc = true;
    c.moistureCouplingEnabled = true;
    c.latentCouplingMode = 1;
    expectTrue(simulation::latentCouplingActive(c), "latent active when humidity+T+moisture on");
    c.latentCouplingMode = 2;
    expectTrue(simulation::latentCouplingActive(c), "latent active for from_phase_change");
    c.humidityCalc = false;
    expectTrue(!simulation::latentCouplingActive(c), "latent inactive when humidity off");
    c.humidityCalc = true;
    c.temperatureCalc = false;
    expectTrue(!simulation::latentCouplingActive(c), "latent inactive when temperature off");
    c.temperatureCalc = true;
    c.moistureCouplingEnabled = false;
    expectTrue(!simulation::latentCouplingActive(c), "latent inactive when moisture decoupled");
    c.moistureCouplingEnabled = true;
    c.latentCouplingMode = 0;
    expectTrue(!simulation::latentCouplingActive(c), "latent inactive when disabled");
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

    {
        auto j = minimalSimJson();
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.latentCouplingMode == 0, "parser: latent default Disabled");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"latent_coupling_mode", "disabled"}};
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.latentCouplingMode == 0, "parser: disabled");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"latent_coupling_mode", "from_humidity_change"}};
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.latentCouplingMode == 1, "parser: from_humidity_change");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"latent_coupling_mode", "from_phase_change"}};
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.latentCouplingMode == 2, "parser: from_phase_change");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"latent_coupling_mode", "feedback_to_thermal"}};
        const auto c = parseSimulationConstants(j, logs);
        expectTrue(c.latentCouplingMode == 1, "parser: feedback_to_thermal alias");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {
            {"moisture_enabled", false},
            {"latent_coupling_mode", "from_humidity_change"},
        };
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("moisture_enabled") != std::string::npos,
                       "parser: reject message mentions moisture_enabled");
        }
        expectTrue(threw, "parser: reject latent with moisture decoupled");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["calc_flag"] = {{"p", true}, {"t", false}, {"x", true}, {"c", false}};
        j["simulation"]["coupling"] = {{"latent_coupling_mode", "from_humidity_change"}};
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error& e) {
            threw = true;
            expectTrue(std::string(e.what()).find("calc_flag.t") != std::string::npos,
                       "parser: reject message mentions calc_flag.t");
        }
        expectTrue(threw, "parser: reject latent without temperatureCalc");
    }
    {
        auto j = minimalSimJson();
        const auto c = parseSimulationConstants(j, logs);
        expectNear(c.temperatureRelaxation, 1.0, 1e-12, "parser: temperature_relaxation default 1");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"temperature_relaxation", 0.5}};
        const auto c = parseSimulationConstants(j, logs);
        expectNear(c.temperatureRelaxation, 0.5, 1e-12, "parser: temperature_relaxation 0.5");
    }
    {
        auto j = minimalSimJson();
        j["simulation"]["coupling"] = {{"temperature_relaxation", 0.0}};
        bool threw = false;
        try {
            (void)parseSimulationConstants(j, logs);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expectTrue(threw, "parser: reject temperature_relaxation 0");
    }
}

} // namespace

int main() {
    testInnerCouplingBreakNoNeed();
    testInnerCouplingConvergedOnSecond();
    testInnerCouplingContinueOnFirst();
    testInnerCouplingAllowFirstWhenMinOne();
    testInnerCouplingPressureFirstFail();
    testInnerCouplingMaxIterations();
    testTemperatureRelaxationAlpha();
    testUnphysicalTemperatureDetection();
    testUnphysicalHumidityDetection();
    testHumidityCouplingFlags();
    testAirconDecideActions();
    testOuterMaxIterationsThrow();
    testOuterAcceptDoesNotThrow();
    testLatentModeResolve();
    testLatentFromHumidityChangeKnownDx();
    testLatentCouplingActiveRequiresHumidity();
    testEffectiveIterationFallback();
    testParserPositiveIntegerIterations();

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[DONE] failures=" << g_failures << "\n";
    return 1;
}

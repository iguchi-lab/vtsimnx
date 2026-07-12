#pragma once

#include "simulation_runner_helpers.h"
#include "core/humidity/humidity_solver.h"

#include <cstddef>
#include <ostream>
#include <string>

namespace simulation {
namespace detail {

enum class InnerCouplingAction {
    Continue,
    BreakNoNeed,
    BreakConverged,
    ThrowPressureNonConvergence,
    ThrowMaxIteration,
};

struct InnerCouplingEval {
    InnerCouplingAction action = InnerCouplingAction::Continue;
    CoupledDelta delta{};
    std::string dominant = "none";
    double pressureTol = 0.0;
    double temperatureTol = 0.0;
    double humidityTol = 0.0;
    double latentTol = 0.0;
};

// minCouplingIterations: 通常1、モード切替等では2。
InnerCouplingEval evaluateInnerCoupling(const SimulationConstants& constants,
                                        bool humidityActive,
                                        bool latentActive,
                                        std::size_t coupledIter,
                                        std::size_t minCouplingIterations,
                                        const CoupledDelta& delta,
                                        bool pressureConvergedAfterFirstSolve);

// ---- ログ組み立て（計算本体から分離） ----
void logHumiditySolverNotConverged(std::ostream& logs,
                                   bool logEnabled,
                                   const core::humidity::HumiditySolveStats& stats);

void logPressureFallbackStop(std::ostream& logs, bool logEnabled);

void logInnerCouplingNotNeeded(std::ostream& logs, bool logEnabled);

void logInnerCouplingDelta(std::ostream& logs,
                           bool logEnabled,
                           const CoupledDelta& delta,
                           double latentAppliedW,
                           const core::humidity::HumiditySolveStats& humidityStats);

void logInnerCouplingConverged(std::ostream& logs, bool logEnabled, std::size_t coupledIter);

void logInnerCouplingMaxIteration(std::ostream& logs,
                                  bool logEnabled,
                                  std::size_t coupledIter,
                                  const InnerCouplingEval& eval,
                                  double latentAppliedW,
                                  const core::humidity::HumiditySolveStats& humidityStats);

void logAirconRecompute(std::ostream& logs, bool logEnabled);

void logThermalNotConverged(std::ostream& logs,
                            bool logEnabled,
                            const std::string& method,
                            double rmseBalance,
                            double maxBalance,
                            int loopIndex1Based);

void logOuterLoopConverged(std::ostream& logs, bool logEnabled, int loopIndex1Based);

void logTimestepFinished(std::ostream& logs, bool logEnabled, int totalIterations);

// 外側空調ループが上限内に Accept できなかった場合に throw する。
void ensureOuterAirconLoopConverged(bool outerLoopConverged);

} // namespace detail
} // namespace simulation

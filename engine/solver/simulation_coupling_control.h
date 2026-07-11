#pragma once

#include "simulation_runner_helpers.h"
#include "core/humidity/humidity_solver.h"

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
};

// 内側連成の収束・打ち切り判定（ログは書かない）
InnerCouplingEval evaluateInnerCoupling(const SimulationConstants& constants,
                                        bool humidityActive,
                                        int coupledIter,
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

void logInnerCouplingConverged(std::ostream& logs, bool logEnabled, int coupledIter);

void logInnerCouplingMaxIteration(std::ostream& logs,
                                  bool logEnabled,
                                  int coupledIter,
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

} // namespace detail
} // namespace simulation

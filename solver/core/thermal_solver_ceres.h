#pragma once

#include "vtsim_solver.h"

#include <ceres/ceres.h>
#include <functional>
#include <ostream>
#include <string>

namespace ThermalSolverCeres {

bool runThermalSolverTrial(const std::string& startLog,
                           const std::string& successLog,
                           ceres::Problem& problem,
                           ceres::Solver::Summary& summary,
                           double successTolerance,
                           const std::function<void(ceres::Solver::Options&)>& configureOptions,
                           std::ostream& logFile);

void runThermalSolvers(const SimulationConstants& constants,
                       ceres::Problem& problem,
                       ceres::Solver::Summary& summary,
                       std::ostream& logFile);

} // namespace ThermalSolverCeres



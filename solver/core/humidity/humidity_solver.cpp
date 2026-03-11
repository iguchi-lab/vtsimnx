#include "core/humidity/humidity_solver.h"

#include "core/humidity/humidity_coupling.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"

#include <vector>

namespace core::humidity {

HumiditySolveStats updateHumidityIfEnabled(const SimulationConstants& constants,
                                           VentilationNetwork& ventNetwork,
                                           ThermalNetwork& thermalNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logs,
                                           TimingList& timings,
                                           const std::string& meta) {
    (void)logs;
    HumiditySolveStats stats{};
    if (!constants.humidityCalc) return stats;

    ScopedTimer timer(timings, "humidity_update", meta);

    auto& tGraph = thermalNetwork.getGraph();
    auto& vGraph = ventNetwork.getGraph();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return stats;

    (void)flowRates; // エッジ直接走査方式に統一したため FlowRateMap は不使用
    HumidityNetworkTerms terms;
    thermalNetwork.buildHumidityNetworkTerms(ventNetwork, terms);
    stats.activeVertices = static_cast<int>(terms.updateVertices.size());
    if (stats.activeVertices == 0) return stats;

    std::vector<double> xOld;
    std::vector<double> xNew;
    initializeHumidityState(tGraph, xOld, xNew);
    const SolveStats solve = solveHumidityImplicitStep(
        tGraph,
        terms,
        dt,
        constants.humiditySolverTolerance,
        xNew,
        xOld);
    applyHumidityStateToGraphs(tGraph, vGraph, vKeyToV, terms.updateVertices, xNew);
    stats.updated = true;
    stats.iterations = solve.iterations;
    stats.finalMaxDiff = solve.finalMaxDiff;
    stats.converged = solve.converged;
    return stats;
}

} // namespace core::humidity


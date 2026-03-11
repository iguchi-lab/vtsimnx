#include "core/humidity/humidity_solver.h"

#include "core/humidity/humidity_coupling.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"

#include <vector>

namespace core::humidity {

void updateHumidityIfEnabled(const SimulationConstants& constants,
                             VentilationNetwork& ventNetwork,
                             ThermalNetwork& thermalNetwork,
                             const FlowRateMap& flowRates,
                             std::ostream& logs,
                             TimingList& timings,
                             const std::string& meta) {
    (void)logs;
    if (!constants.humidityCalc) return;

    ScopedTimer timer(timings, "humidity_update", meta);

    auto& tGraph = thermalNetwork.getGraph();
    auto& vGraph = ventNetwork.getGraph();
    const auto& tKeyToV = thermalNetwork.getKeyToVertex();
    const auto& vKeyToV = ventNetwork.getKeyToVertex();

    const double dt = static_cast<double>(constants.timestep);
    if (!(dt > 0.0)) return;

    (void)flowRates; // エッジ直接走査方式に統一したため FlowRateMap は不使用
    NetworkTerms terms;
    buildHumidityNetworkTerms(vGraph, tGraph, tKeyToV, terms);

    std::vector<double> xOld;
    std::vector<double> xNew;
    initializeHumidityState(tGraph, xOld, xNew);
    solveHumidityImplicitStep(tGraph, terms, dt, xNew, xOld);
    applyHumidityStateToGraphs(tGraph, vGraph, vKeyToV, terms.updateVertices, xNew);
}

} // namespace core::humidity


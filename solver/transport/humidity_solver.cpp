#include "transport/humidity_solver.h"
#include "core/humidity/humidity_solver.h"

namespace transport {

void updateHumidityIfEnabled(const SimulationConstants& constants,
                             VentilationNetwork& ventNetwork,
                             ThermalNetwork& thermalNetwork,
                             const FlowRateMap& flowRates,
                             std::ostream& logs,
                             TimingList& timings,
                             const std::string& meta) {
    // 互換レイヤ: transport API は維持しつつ、実体は core/humidity へ委譲する。
    core::humidity::updateHumidityIfEnabled(constants, ventNetwork, thermalNetwork, flowRates, logs, timings, meta);
}

} // namespace transport



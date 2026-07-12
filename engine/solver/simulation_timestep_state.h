#pragma once

#include "simulation_runner_helpers.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"

#include <vector>

namespace simulation {
namespace detail {

struct TimestepInitialState {
    std::vector<double> humidityX; // xPrev
    std::vector<double> moistureW; // wPrev
};

struct CouplingSnapshot {
    std::vector<double> pressure;
    std::vector<double> temperature;
    std::vector<double> humidity;
    std::vector<double> heatSource;       // scheduled (+ sensible) base for restore
    std::vector<double> latentHeatSource; // humidity latent [W] by vertex
};

struct SharedNodeStateArgs {
    Graph& nodeGraph;
    ConstNodeStateView nodeState;
};

inline SharedNodeStateArgs makeSharedNodeStateArgs(ThermalNetwork& thermalNetwork) {
    return SharedNodeStateArgs{
        thermalNetwork.getGraph(),
        static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView(),
    };
}

inline TimestepInitialState captureTimestepInitialState(ThermalNetwork& thermalNetwork,
                                                        bool humidityCalc) {
    TimestepInitialState initial;
    if (humidityCalc) {
        captureXPrevByVertex(thermalNetwork.getGraph(), initial.humidityX);
        captureWPrevByVertex(thermalNetwork.getGraph(), initial.moistureW);
    }
    return initial;
}

inline void captureCouplingPrevState(CouplingSnapshot& snap,
                                     VentilationNetwork& ventNetwork,
                                     ThermalNetwork& thermalNetwork,
                                     const SimulationConstants& constants,
                                     bool humidityActive) {
    if (constants.pressureCalc) {
        snap.pressure = ventNetwork.collectPressureValues();
    }
    if (constants.temperatureCalc) {
        capturePrevTempsByVertex(thermalNetwork.getGraph(), snap.temperature);
    }
    if (humidityActive) {
        capturePrevHumidityByVertex(thermalNetwork.getGraph(), snap.humidity);
    }
}

} // namespace detail
} // namespace simulation

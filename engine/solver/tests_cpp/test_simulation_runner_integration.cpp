#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "aircon/aircon_controller.h"
#include "network/contaminant_network.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_aircon_iteration.h"
#include "simulation_error.h"
#include "simulation_runner.h"
#include "types/common_types.h"
#include "vtsim_solver.h"
#include "vtsimnx_solver_timing.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

std::optional<simulation::AirconIterationAction> alwaysRecomputeForCapacity() {
    return simulation::AirconIterationAction::RecomputeForCapacity;
}

} // namespace

int main() {
    SimulationConstants constants{};
    constants.logVerbosity = 0;
    constants.pressureCalc = false;
    constants.temperatureCalc = false;
    constants.humidityCalc = false;
    constants.concentrationCalc = false;
    constants.moistureCouplingEnabled = false;
    constants.maxInnerIterations = 5;
    constants.maxCouplingIterations = 5;
    constants.maxAirconControlIterations = 3;
    constants.ventilationTolerance = 1e-3;
    constants.thermalTolerance = 1e-3;
    constants.convergenceTolerance = 1e-3;

    VentilationNetwork vent;
    ThermalNetwork thermal;
    HumidityNetwork humidity;
    ContaminantNetwork contaminant;
    AirconController aircon;
    std::ostringstream logs;
    TimingList timings;
    TimestepResult result;

    // 空トポロジでも runSimulation 全体を通し、外側上限で必ず失敗することを確認する。
    vent.buildFromData({}, {}, constants, logs);
    thermal.buildFromData({}, {}, {}, constants, logs);

    simulation::test_hooks::ScopedAirconIterationOverride hook(alwaysRecomputeForCapacity);

    bool threw = false;
    try {
        runSimulation(vent, thermal, humidity, contaminant, aircon, constants, result, logs, timings, "integ");
    } catch (const simulation::Error& e) {
        threw = true;
        expectTrue(e.code() == simulation::ErrorCode::AirconMaxIterations,
                   "runSimulation: AirconMaxIterations code");
        expectTrue(std::string(e.what()).find("Aircon control did not converge") != std::string::npos,
                   "runSimulation: message");
    } catch (const std::exception& e) {
        fail(std::string("unexpected exception: ") + e.what());
    }

    expectTrue(threw, "runSimulation must throw AirconMaxIterations via ensureOuterAirconLoopConverged");

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[DONE] failures=" << g_failures << "\n";
    return 1;
}

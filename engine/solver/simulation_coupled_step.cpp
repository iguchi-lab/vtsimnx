#include "simulation_coupled_step.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_error.h"
#include "utils/utils.h"

#include <memory>

CoupledStepData performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                                              ThermalNetwork& thermalNetwork,
                                              const SimulationConstants& constants,
                                              std::ostream& logs,
                                              TimingList& timings,
                                              const std::string& meta) {
    const bool logEnabled = (constants.logVerbosity > 0);
    CoupledStepData step;

    // 換気計算
    if (constants.pressureCalc) {
        std::unique_ptr<ScopedLogSection> pressureScope;
        if (logEnabled) pressureScope = std::make_unique<ScopedLogSection>(logs, "圧力計算");
        PressureSolveResult pressureResult;
        {
            ScopedTimer timer(timings, "pressure_solve_iteration", meta);
            pressureResult = ventNetwork.solvePressureDetailed(constants, logs);
        }
        step.pressureMap = std::move(pressureResult.pressures);
        step.flowRates = std::move(pressureResult.flows);
        step.flowBalance = std::move(pressureResult.balances);

        // 不採用解はグラフへ反映せず、熱計算にも進まない（ThermalNotConverged で隠さない）
        if (!pressureResult.accepted) {
            throw simulation::Error(
                simulation::ErrorCode::PressureNotConverged,
                std::string("Pressure solver did not accept solution during coupled step (method=") +
                    (pressureResult.method.empty() ? "none" : pressureResult.method) + ")");
        }
        ventNetwork.applySolveResults(step.pressureMap, step.flowRates);
    }

    // 熱計算
    if (constants.temperatureCalc) {
        // pressureCalc=false の場合でも fixed_flow 等で flow_rate が入るため、移流用に同期する
        // pressureCalc=true の場合も換気計算結果を熱回路網に同期する
        thermalNetwork.syncFlowRatesFromVentilationNetwork(ventNetwork);
        std::unique_ptr<ScopedLogSection> thermalScope;
        if (logEnabled) thermalScope = std::make_unique<ScopedLogSection>(logs, "熱計算");
        {
            ScopedTimer timer(timings, "thermal_solve_iteration", meta);
            thermalNetwork.solveTemperature(constants, logs);
        }
        // 内側連成を古い温度で続けないよう、熱ソルバ失敗/未収束はここで打ち切る。
        if (!thermalNetwork.getLastThermalConverged()) {
            throw simulation::Error(
                simulation::ErrorCode::ThermalNotConverged,
                std::string("Thermal solver did not converge during coupled step (method=") +
                    thermalNetwork.getLastThermalMethod() + ")");
        }

        // pressureCalc=false の場合、換気側で温度（密度）を参照する計算が走らないため更新不要
        if (constants.pressureCalc) {
            ventNetwork.syncTemperaturesFromThermalNetwork(thermalNetwork);
        }
    }

    return step;
}

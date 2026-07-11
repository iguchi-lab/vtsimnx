#include "simulation_coupled_step.h"

#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <memory>

CoupledStepData performCoupledStepCalculation(VentilationNetwork& ventNetwork,
                                              ThermalNetwork& thermalNetwork,
                                              const SimulationConstants& constants,
                                              std::ostream& logs,
                                              int& totalIterations,
                                              TimingList& timings,
                                              const std::string& meta) {
    (void)totalIterations; // runSimulation 側で反復回数を管理する
    const bool logEnabled = (constants.logVerbosity > 0);
    CoupledStepData step;

    // 換気計算
    if (constants.pressureCalc) {
        std::unique_ptr<ScopedLogSection> pressureScope;
        if (logEnabled) pressureScope = std::make_unique<ScopedLogSection>(logs, "圧力計算");
        {
            ScopedTimer timer(timings, "pressure_solve_iteration", meta);
            std::tie(step.pressureMap, step.flowRates, step.flowBalance) =
                ventNetwork.solvePressure(constants, logs);
        }
        ventNetwork.applySolveResults(step.pressureMap, step.flowRates);

        // runSimulation 側の1回目チェックと同じ条件で止めたいので、ここでは totalIterations を見ない
        // （未収束フラグは solve 後に network 側に保持される）
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

        // pressureCalc=false の場合、換気側で温度（密度）を参照する計算が走らないため更新不要
        if (constants.pressureCalc) {
            ventNetwork.syncTemperaturesFromThermalNetwork(thermalNetwork);
        }
    }

    return step;
}

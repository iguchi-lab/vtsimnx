#include "simulation_aircon_iteration.h"

#include "aircon/aircon_controller.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"

#include <boost/range/iterator_range.hpp>

AirconIterationResult runAirconIteration(AirconController& airconController,
                                         ThermalNetwork& thermalNetwork,
                                         VentilationNetwork& ventNetwork,
                                         const SimulationConstants& constants,
                                         const FlowRateMap& flowRates,
                                         std::ostream& logs,
                                         int& totalIterations,
                                         TimingList& timings,
                                         const std::string& meta) {
    AirconIterationResult r;

    // 0. DUCT_CENTRAL の処理熱量連動風量を補正（変更が入ったら外側ループをやり直し）
    {
        ScopedTimer timer(timings, "aircon_duct_flow_adjust", meta);
        const bool ductFlowAdjusted = airconController.checkAndAdjustDuctCentralAirflow(
            thermalNetwork, ventNetwork, flowRates, logs);
        if (ductFlowAdjusted) {
            r.action = AirconIterationAction::RecomputeForFlow;
            return r;
        }
    }

    // 1. 現在の温度でエアコン出力を決定し、各ノードの heat_source をリセットする
    bool allAirconControlled = false;
    {
        auto& graph = thermalNetwork.getGraph();
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            graph[v].heat_source = 0.0;
        }

        ScopedTimer timer(timings, "aircon_control", meta);
        allAirconControlled = airconController.controlAllAircons(
            thermalNetwork, constants.thermalTolerance, logs);
    }

    // 2. エアコンが ON の場合、必要に応じて追加の処理（現状は行列側でA案として処理されるため、ここでの heat_source 設定は不要）

    if (!allAirconControlled) {
        r.action = AirconIterationAction::RecomputeForControl;
        return r;
    }

    bool adjustmentMade = false;
    {
        ScopedTimer timer(timings, "aircon_capacity_adjust", meta);
        adjustmentMade = airconController.checkAndAdjustCapacity(
            thermalNetwork, ventNetwork, constants, flowRates, logs, totalIterations);
    }

    if (adjustmentMade) {
        r.action = AirconIterationAction::RecomputeForCapacity;
        return r;
    }

    r.action = AirconIterationAction::Accept;
    return r;
}

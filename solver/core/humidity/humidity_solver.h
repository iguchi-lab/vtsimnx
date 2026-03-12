#pragma once

#include "types/common_types.h"
#include "network/node_state_view.h"
#include "types/graph_types.h"
#include "vtsimnx_solver_timing.h"

class VentilationNetwork;
class HumidityNetwork;

namespace core::humidity {

struct HumiditySolveStats {
    bool updated = false;
    int activeVertices = 0;
    int iterations = 0;
    double finalMaxDiff = 0.0;
    bool converged = true;
};

// 湿度（絶対湿度 x）を 1 タイムステップ分だけ更新する。
// - constants.humidityCalc=false の場合は何もしない。
// - flowRates は現状未使用（換気グラフの枝流量を直接参照）だが、runner との
//   インターフェース互換を保つため受け取る（将来の差分流量連携に備える）。
HumiditySolveStats updateHumidityIfEnabled(const SimulationConstants& constants,
                                           VentilationNetwork& ventNetwork,
                                           Graph& nodeGraph,
                                           ConstNodeStateView nodeState,
                                           HumidityNetwork& humidityNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logs,
                                           TimingList& timings,
                                           const std::string& meta);

} // namespace core::humidity


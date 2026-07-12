#pragma once

#include "types/common_types.h"
#include "network/node_state_view.h"
#include "types/graph_types.h"
#include "vtsimnx_solver_timing.h"

#include <cstddef>
#include <string>
#include <vector>

class VentilationNetwork;
class HumidityNetwork;

namespace simulation {
struct TimestepSolveMetrics;
}

namespace core::humidity {

struct HumiditySolveStats {
    bool updated = false;
    int activeVertices = 0;
    int iterations = 0;
    // ||Ax-b|| / ||b|| （連成の湿度変化量とは別物）
    double finalRelativeResidual = 0.0;
    bool converged = true;
    std::size_t patternAnalyzes = 0;
    std::size_t factorizes = 0;
    std::size_t rhsOnlySolves = 0;
    std::size_t solutionReuse = 0;
    // 水分収支診断: max|storage - (vent+gen+material+aircon)| [kg/s]
    double maxMoistureBalanceResidual = 0.0;
};

// 湿度（絶対湿度 x）を 1 タイムステップ分だけ更新する。
// - constants.humidityCalc=false の場合は何もしない。
// - flowRates は現状未使用（換気グラフの枝流量を直接参照）だが、runner との
//   インターフェース互換を保つため受け取る（将来の差分流量連携に備える）。
// - xN: タイムステップ初期湿度。nullptr の場合は現グラフを x_n として使う（非連成経路）。
// - metrics: 任意。与えられた場合はキャッシュ段カウンタを加算する。
HumiditySolveStats updateHumidityIfEnabled(const SimulationConstants& constants,
                                           VentilationNetwork& ventNetwork,
                                           Graph& nodeGraph,
                                           ConstNodeStateView nodeState,
                                           HumidityNetwork& humidityNetwork,
                                           const FlowRateMap& flowRates,
                                           std::ostream& logs,
                                           TimingList& timings,
                                           const std::string& meta,
                                           const std::vector<double>* xN = nullptr,
                                           simulation::TimestepSolveMetrics* metrics = nullptr);

} // namespace core::humidity

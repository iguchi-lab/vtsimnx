#pragma once

#include "types/common_types.h"
#include "network/node_state_view.h"
#include "types/graph_types.h"
#include "vtsimnx_solver_timing.h"

class VentilationNetwork;
class ContaminantNetwork;

namespace transport {

struct ConcentrationSolveStats {
    bool updated = false;
    int activeVertices = 0;
    bool converged = true;
};

// 濃度（c）を 1 タイムステップ分だけ更新する。
// - 結合した未知濃度は有限体積の連立陰解法。V<=0は定常混合行。
// - 他の未知節点と結合しない有限体積は局所解析解を保持する。
// - HRV内部節点も同時に解き、閉鎖系のΣV*cを保存する。
// - constants.concentrationCalc=false の場合は何もしない。
// - 非有限・負値を含む解は反映せず converged=false。
ConcentrationSolveStats updateConcentrationIfEnabled(const SimulationConstants& constants,
                                                     VentilationNetwork& ventNetwork,
                                                     Graph& nodeGraph,
                                                     ConstNodeStateView nodeState,
                                                     ContaminantNetwork& contaminantNetwork,
                                                     std::ostream& logs,
                                                     TimingList& timings,
                                                     const std::string& meta);

} // namespace transport

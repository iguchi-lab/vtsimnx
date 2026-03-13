#pragma once

#include "types/common_types.h"
#include "network/node_state_view.h"
#include "types/graph_types.h"
#include "vtsimnx_solver_timing.h"

class VentilationNetwork;
class ContaminantNetwork;

namespace transport {

// 濃度（c）を 1 タイムステップ分だけ更新する。
// - old_vtsim 互換の式:
//   dc/dt = k1 - k2*c,  c(t+dt) = (c - k)*exp(-k2*dt) + k,  k=k1/k2
//   k2 = beta + Σ(outflow)/V
//   k1 = m/V + Σ(inflow)*(1-eta)*c_src/V
// - constants.concentrationCalc=false の場合は何もしない。
void updateConcentrationIfEnabled(const SimulationConstants& constants,
                                  VentilationNetwork& ventNetwork,
                                  Graph& nodeGraph,
                                  ConstNodeStateView nodeState,
                                  ContaminantNetwork& contaminantNetwork,
                                  std::ostream& logs,
                                  TimingList& timings,
                                  const std::string& meta);

} // namespace transport



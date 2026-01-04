#include "core/thermal/thermal_direct_internal.h"

namespace ThermalSolverLinearDirect::detail {

TopologyCache g_topologyCache;
SparseLUCache g_sparseLuCache;
SparseCholeskyCache g_cholCache;
DirectTStats g_directTStats;
std::uint64_t s_lastCoeffSig = 0;

} // namespace ThermalSolverLinearDirect::detail



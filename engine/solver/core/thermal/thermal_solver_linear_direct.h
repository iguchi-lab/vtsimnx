#pragma once

#include "../../vtsim_solver.h"
#include <cstdint>
#include <ostream>

// 前方宣言
class ThermalNetwork;

namespace ThermalSolverLinearDirect {

struct DirectTCacheStats {
    std::uint64_t calls = 0;
    std::uint64_t coeffSigChanged = 0;
    std::uint64_t rhsOnlyBuild = 0;
    std::uint64_t fullBuild = 0;
    std::uint64_t solveCached = 0;
    std::uint64_t solveFull = 0;
};

// 熱計算（温度に関して線形を前提）を「絶対温度」AT=b の形で疎直接法で解く（LLT/LDLT/SparseLU）
void solveTemperaturesLinearDirect(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile);

// テスト/診断用（キャッシュ挙動の検証に使用）
DirectTCacheStats getDirectTCacheStats();
void resetDirectTCacheStats();

} // namespace ThermalSolverLinearDirect



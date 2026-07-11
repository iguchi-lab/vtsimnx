#pragma once

#include "../../vtsim_solver.h"
#include <cstdint>
#include <ostream>

// 前方宣言
class ThermalNetwork;

namespace ThermalSolverLinearDirect {

namespace detail {
struct DirectTSolverContext;
}

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

// 明示コンテキスト版（並列化・テスト用）。ctx に LU/トポロジ/統計を保持する。
void solveTemperaturesLinearDirect(
    ThermalNetwork& network,
    const SimulationConstants& constants,
    std::ostream& logFile,
    detail::DirectTSolverContext& ctx);

// テスト/診断用（キャッシュ挙動の検証に使用）
DirectTCacheStats getDirectTCacheStats();
DirectTCacheStats getDirectTCacheStats(detail::DirectTSolverContext& ctx);
void resetDirectTCacheStats();
// 既定コンテキストを明示的に初期化（キャッシュ破棄）する
void resetDirectTSolverContext();

} // namespace ThermalSolverLinearDirect

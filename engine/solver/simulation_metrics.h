#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace simulation {

// タイムステップ単位の連成・ソルバ計測。
struct TimestepSolveMetrics {
    std::size_t outerIterations = 0;
    std::size_t coupledIterations = 0; // 内側連成の累積回数（外側をまたぐ）
    std::size_t pressureSolveCalls = 0;
    std::size_t pressureCeresIterations = 0;
    std::size_t pressureFallbackCount = 0;
    std::size_t thermalRhsOnlyBuilds = 0;
    std::size_t thermalFullBuilds = 0;
    std::size_t humidityPatternAnalyzes = 0;
    std::size_t humidityFactorizes = 0;
    std::size_t humidityRhsOnlySolves = 0;
    std::size_t humiditySolutionReuse = 0;
    std::size_t airconFlowAdjustRecalc = 0;
    std::size_t airconOnOffRecalc = 0;
    std::size_t airconCapacityRecalc = 0;
    double pressureMs = 0.0;
    double thermalMs = 0.0;
    double humidityMs = 0.0;
    double airconMs = 0.0;

    void reset() { *this = TimestepSolveMetrics{}; }

    nlohmann::json toJson() const {
        return nlohmann::json{
            {"outer_iterations", outerIterations},
            {"coupled_iterations", coupledIterations},
            {"pressure_solve_calls", pressureSolveCalls},
            {"pressure_ceres_iterations", pressureCeresIterations},
            {"pressure_fallback_count", pressureFallbackCount},
            {"thermal_rhs_only_builds", thermalRhsOnlyBuilds},
            {"thermal_full_builds", thermalFullBuilds},
            {"humidity_pattern_analyzes", humidityPatternAnalyzes},
            {"humidity_factorizes", humidityFactorizes},
            {"humidity_rhs_only_solves", humidityRhsOnlySolves},
            {"humidity_solution_reuse", humiditySolutionReuse},
            {"aircon_flow_adjust_recalc", airconFlowAdjustRecalc},
            {"aircon_on_off_recalc", airconOnOffRecalc},
            {"aircon_capacity_recalc", airconCapacityRecalc},
            {"pressure_ms", pressureMs},
            {"thermal_ms", thermalMs},
            {"humidity_ms", humidityMs},
            {"aircon_ms", airconMs},
        };
    }
};

// meta に outerIter / coupledIter を必ず付ける。
inline std::string appendLoopMeta(std::string_view base,
                                  int outerIter1Based,
                                  std::optional<int> coupledIter1Based = std::nullopt) {
    std::string out(base);
    if (!out.empty()) out.push_back(',');
    out += "outerIter=";
    out += std::to_string(outerIter1Based);
    if (coupledIter1Based.has_value()) {
        out += ",coupledIter=";
        out += std::to_string(*coupledIter1Based);
    }
    return out;
}

} // namespace simulation

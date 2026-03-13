#pragma once

#include <string>
#include <ostream>
#include <nlohmann/json.hpp>
#include "../vtsim_solver.h"

// JSON の simulation 定数を解析して SimulationConstants を返し、
// ログ出力ストリーム（改行区切り）に追記する。
SimulationConstants parseSimulationConstants(const nlohmann::json& config,
                                             std::ostream& logs);



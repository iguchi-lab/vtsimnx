#pragma once

#include <string>
#include <vector>
#include <ostream>
#include <nlohmann/json.hpp>
#include "../vtsim_solver.h"

// 換気ブランチ配列を JSON から読み取り、timestep に応じた current_* を設定して返す
std::vector<EdgeProperties> parseVentilationBranches(const nlohmann::json& config,
                                                     std::ostream& logs,
                                                     long timestep);

// 熱ブランチ配列を JSON から読み取り、timestep に応じた current_* を設定して返す
std::vector<EdgeProperties> parseThermalBranches(const nlohmann::json& config,
                                                 std::ostream& logs,
                                                 long timestep);



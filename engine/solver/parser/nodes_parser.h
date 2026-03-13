#pragma once

#include <string>
#include <vector>
#include <ostream>
#include <nlohmann/json.hpp>
#include "../vtsim_solver.h"

// ノード配列を JSON から読み取り、timestep に応じた current_* を設定して返す
std::vector<VertexProperties> parseNodes(const nlohmann::json& config,
                                         std::ostream& logs,
                                         long timestep);



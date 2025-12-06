#include "nodes_parser.h"
#include "parser_utils.h"
#include <stdexcept>
#include <string>

using nlohmann::json;

namespace { }

std::vector<VertexProperties> parseNodes(const json& config, std::ostream& logs, long timestep) {
    std::vector<VertexProperties> nodes;

    if (!config.contains("nodes") || !config["nodes"].is_array()) {
        logs << "--[WARN] nodes 配列が見つかりません。\n";
        return nodes;
    }

    const int verbosity = parser_utils::readVerbosity(config);
    const size_t total = config["nodes"].size();
    size_t index = 0;
    for (const auto& nodeJson : config["nodes"]) {
        ++index;
        VertexProperties node{};
        const std::string nodePrefix = "nodes[" + std::to_string(index-1) + "]";

        // 必須/基本フィールド
        if (nodeJson.contains("key") && !nodeJson["key"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].key must be string");
        if (nodeJson.contains("name") && !nodeJson["name"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].name must be string");
        if (nodeJson.contains("type") && !nodeJson["type"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].type must be string");
        if (nodeJson.contains("subtype") && !nodeJson["subtype"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].subtype must be string");
        if (nodeJson.contains("comment") && !nodeJson["comment"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].comment must be string");
        if (nodeJson.contains("ref_node") && !nodeJson["ref_node"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].ref_node must be string");
        if (nodeJson.contains("in_node") && !nodeJson["in_node"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].in_node must be string");
        if (nodeJson.contains("set_node") && !nodeJson["set_node"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].set_node must be string");
        if (nodeJson.contains("outside_node") && !nodeJson["outside_node"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].outside_node must be string");
        if (nodeJson.contains("model") && !nodeJson["model"].is_string())
            throw std::runtime_error("nodes[" + std::to_string(index-1) + "].model must be string");
        if (nodeJson.contains("key"))    node.key  = nodeJson["key"].get<std::string>();
        if (nodeJson.contains("name"))   node.name = nodeJson["name"].get<std::string>();
        if (nodeJson.contains("type"))   node.type = nodeJson["type"].get<std::string>();
        if (nodeJson.contains("subtype")) node.subtype = nodeJson["subtype"].get<std::string>();
        if (nodeJson.contains("comment")) node.comment = nodeJson["comment"].get<std::string>();
        if (nodeJson.contains("ref_node")) node.ref_node = nodeJson["ref_node"].get<std::string>();
        if (nodeJson.contains("in_node")) node.in_node = nodeJson["in_node"].get<std::string>();
        if (nodeJson.contains("set_node")) node.set_node = nodeJson["set_node"].get<std::string>();
        if (nodeJson.contains("outside_node")) node.outside_node = nodeJson["outside_node"].get<std::string>();
        if (nodeJson.contains("model")) node.model = nodeJson["model"].get<std::string>();

        // 計算フラグ
        auto requireBoolean = [&](const char* key) {
            if (!nodeJson[key].is_boolean()) {
                throw std::runtime_error(nodePrefix + "." + key + " must be boolean");
            }
            return nodeJson[key].get<bool>();
        };
        if (nodeJson.contains("calc_p")) node.calc_p = requireBoolean("calc_p");
        if (nodeJson.contains("calc_t")) node.calc_t = requireBoolean("calc_t");
        if (nodeJson.contains("calc_x")) node.calc_x = requireBoolean("calc_x");
        if (nodeJson.contains("calc_c")) node.calc_c = requireBoolean("calc_c");

        // 時系列ベクトル（配列/単一値の両対応）
        auto readSeries = [&](const char* field, std::vector<double>& storage, double fallback) -> double {
            return parser_utils::readScalarOrSeries<double>(
                nodeJson[field],
                storage,
                static_cast<size_t>(timestep),
                fallback,
                nodePrefix + "." + field);
        };
        if (nodeJson.contains("p")) {
            node.current_p = readSeries("p", node.p, 0.0);
        }
        if (nodeJson.contains("t")) {
            node.current_t = readSeries("t", node.t, 0.0);
        }
        if (nodeJson.contains("x")) {
            node.current_x = readSeries("x", node.x, 0.0);
        }
        if (nodeJson.contains("c")) {
            node.current_c = readSeries("c", node.c, 0.0);
        }
        if (nodeJson.contains("pre_temp")) {
            node.current_pre_temp = readSeries("pre_temp", node.pre_temp, node.current_pre_temp);
        }

        // モード（配列/単一値の両対応）
        if (nodeJson.contains("mode")) {
            const auto& mj = nodeJson["mode"];
            if (mj.is_array()) {
                for (const auto& v : mj) {
                    if (!v.is_string()) {
                        throw std::runtime_error("nodes[" + std::to_string(index-1) + "].mode must be array<string>");
                    }
                }
                node.mode = mj.get<std::vector<std::string>>();
                if (!node.mode.empty()) {
                    node.current_mode = timestep < static_cast<long>(node.mode.size())
                        ? node.mode[static_cast<size_t>(timestep)]
                        : node.mode.back();
                }
            } else if (mj.is_string()) {
                node.current_mode = mj.get<std::string>();
            } else {
                throw std::runtime_error("nodes[" + std::to_string(index-1) + "].mode must be string or array<string>");
            }
        }

        // 容積
        if (nodeJson.contains("v")) {
            if (!nodeJson["v"].is_number()) {
                throw std::runtime_error("nodes[" + std::to_string(index-1) + "].v must be number");
            }
            node.v = nodeJson["v"].get<double>();
        }

        // エアコン仕様原本
        if (nodeJson.contains("ac_spec")) node.ac_spec = nodeJson["ac_spec"];

        // 型が aircon の場合、仕様を初期化
        if (node.type == "aircon") {
            node.initializeAirconSpec();
        }

        // timestep に応じた更新ヘルパ（保守目的）
        node.updateForTimestep(timestep);

        nodes.push_back(std::move(node));
        if (verbosity >= 2) {
            logs << "---ノード: "
                 << index << "/" << total << ": "
                 << nodes.back().key << ", "
                 << nodes.back().name << ", "
                 << nodes.back().type << ", "
                 << nodes.back().subtype << "\n";
        }
    }

    logs << "--全てのノードデータを読み込みました: " << nodes.size() << "個\n";
    return nodes;
}

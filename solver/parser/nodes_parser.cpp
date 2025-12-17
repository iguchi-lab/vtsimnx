#include "nodes_parser.h"
#include "parser_utils.h"
#include "utils/utils.h"
#include <stdexcept>
#include <string>
#include <sstream>

using nlohmann::json;

namespace { }

std::vector<VertexProperties> parseNodes(const json& config, std::ostream& logs, long timestep) {
    std::vector<VertexProperties> nodes;

    if (!config.contains("nodes") || !config["nodes"].is_array()) {
        writeLog(logs, "  [WARN] nodes 配列が見つかりません。");
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
        parser_utils::checkStringIfPresent(nodeJson, "key", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "name", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "type", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "subtype", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "comment", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "ref_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "in_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "set_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "outside_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "model", nodePrefix);
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
        if (nodeJson.contains("calc_p")) node.calc_p = parser_utils::getBooleanIfPresent(nodeJson, "calc_p", nodePrefix, false);
        if (nodeJson.contains("calc_t")) node.calc_t = parser_utils::getBooleanIfPresent(nodeJson, "calc_t", nodePrefix, false);
        if (nodeJson.contains("calc_x")) node.calc_x = parser_utils::getBooleanIfPresent(nodeJson, "calc_x", nodePrefix, false);
        if (nodeJson.contains("calc_c")) node.calc_c = parser_utils::getBooleanIfPresent(nodeJson, "calc_c", nodePrefix, false);

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
            node.v = parser_utils::getNumberIfPresent(nodeJson, "v", nodePrefix, 0.0);
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
            std::ostringstream oss;
            oss << "    ノード: "
                << index << "/" << total << ": "
                << nodes.back().key << ", "
                << nodes.back().name << ", "
                << nodes.back().type << ", "
                << nodes.back().subtype;
            writeLog(logs, oss.str());
        }
    }

    {
        std::ostringstream oss;
        oss << "  全てのノードデータを読み込みました: " << nodes.size() << "個";
        writeLog(logs, oss.str());
    }
    return nodes;
}

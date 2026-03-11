#include "nodes_parser.h"
#include "parser_utils.h"
#include "utils/utils.h"
#include <stdexcept>
#include <string>
#include <sstream>

using nlohmann::json;

namespace {

static inline std::string modeNumberToString(int code, const std::string& nodePrefix) {
    // user-specified mapping:
    // 0: 停止 / 1: 暖房 / 2: 冷房 / 3: 自動
    switch (code) {
        case 0: return "OFF";
        case 1: return "HEATING";
        case 2: return "COOLING";
        case 3: return "AUTO";
        default:
            throw std::runtime_error(nodePrefix + ".mode numeric code must be 0(OFF)/1(HEATING)/2(COOLING)/3(AUTO), got " + std::to_string(code));
    }
}

static inline std::string parseModeValueToString(const json& v, const std::string& nodePrefix) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return modeNumberToString(v.get<int>(), nodePrefix);
    if (v.is_number()) return modeNumberToString(static_cast<int>(v.get<double>()), nodePrefix);
    throw std::runtime_error(nodePrefix + ".mode must be string/number or array<string|number>");
}

} // namespace

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
        parser_utils::requireString(nodeJson, "key", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "name", nodePrefix);
        parser_utils::requireString(nodeJson, "type", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "subtype", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "comment", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "ref_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "in_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "set_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "outside_node", nodePrefix);
        parser_utils::checkStringIfPresent(nodeJson, "model", nodePrefix);
        parser_utils::checkNumberIfPresent(nodeJson, "moisture_capacity", nodePrefix);
        node.key = nodeJson["key"].get<std::string>();
        if (nodeJson.contains("name"))   node.name = nodeJson["name"].get<std::string>();
        node.type = nodeJson["type"].get<std::string>();
        if (nodeJson.contains("subtype")) node.subtype = nodeJson["subtype"].get<std::string>();
        if (nodeJson.contains("comment")) node.comment = nodeJson["comment"].get<std::string>();
        if (nodeJson.contains("ref_node")) node.ref_node = nodeJson["ref_node"].get<std::string>();
        if (nodeJson.contains("in_node")) node.in_node = nodeJson["in_node"].get<std::string>();
        if (nodeJson.contains("set_node")) node.set_node = nodeJson["set_node"].get<std::string>();
        if (nodeJson.contains("outside_node")) node.outside_node = nodeJson["outside_node"].get<std::string>();
        if (nodeJson.contains("model")) node.model = nodeJson["model"].get<std::string>();

        // 計算フラグ
        const bool hasCalcP = nodeJson.contains("calc_p");
        const bool hasCalcT = nodeJson.contains("calc_t");
        const bool hasCalcX = nodeJson.contains("calc_x");
        const bool hasCalcC = nodeJson.contains("calc_c");
        if (hasCalcP) node.calc_p = parser_utils::getBooleanIfPresent(nodeJson, "calc_p", nodePrefix, false);
        if (hasCalcT) node.calc_t = parser_utils::getBooleanIfPresent(nodeJson, "calc_t", nodePrefix, false);
        if (hasCalcX) node.calc_x = parser_utils::getBooleanIfPresent(nodeJson, "calc_x", nodePrefix, false);
        if (hasCalcC) node.calc_c = parser_utils::getBooleanIfPresent(nodeJson, "calc_c", nodePrefix, false);

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
        if (nodeJson.contains("beta")) {
            node.current_beta = readSeries("beta", node.beta, 0.0);
        }
        if (nodeJson.contains("w")) {
            node.current_w = readSeries("w", node.w, 0.0);
        }
        if (nodeJson.contains("pre_temp")) {
            node.current_pre_temp = readSeries("pre_temp", node.pre_temp, node.current_pre_temp);
        }

        // モード（配列/単一値の両対応）
        if (nodeJson.contains("mode")) {
            const auto& mj = nodeJson["mode"];
            if (mj.is_array()) {
                node.mode.clear();
                node.mode.reserve(mj.size());
                for (const auto& v : mj) {
                    node.mode.push_back(parseModeValueToString(v, nodePrefix));
                }
                if (!node.mode.empty()) {
                    const size_t ts = (timestep < 0) ? 0u : static_cast<size_t>(timestep);
                    node.current_mode = (ts < node.mode.size()) ? node.mode[ts] : node.mode.back();
                }
            } else if (mj.is_string()) {
                node.current_mode = mj.get<std::string>();
            } else if (mj.is_number()) {
                node.current_mode = parseModeValueToString(mj, nodePrefix);
            } else {
                throw std::runtime_error("nodes[" + std::to_string(index-1) + "].mode must be string/number or array<string|number>");
            }
        }

        // 容積
        if (nodeJson.contains("v")) {
            node.v = parser_utils::getNumberIfPresent(nodeJson, "v", nodePrefix, 0.0);
        }
        if (nodeJson.contains("moisture_capacity")) {
            node.moisture_capacity =
                parser_utils::getNumberIfPresent(nodeJson, "moisture_capacity", nodePrefix, 0.0);
        }

        // エアコン仕様原本
        if (nodeJson.contains("ac_spec")) node.ac_spec = nodeJson["ac_spec"];

        // 型が aircon の場合、モデル未指定時は RAC をデフォルトにし、仕様を初期化
        if (node.type == "aircon") {
            if (node.model.empty()) {
                node.model = "RAC";
                if (verbosity >= 1) {
                    writeLog(logs, "  [INFO] aircon node: model が未指定のため RAC をデフォルト適用しました: key=" + node.key);
                }
            }
            node.initializeAirconSpec();
        }

        // 重要: エアコンノード（吹出側）は A案で「未知温度」として解く必要がある。
        // 入力で calc_t が省略されるケースがあるため、その場合はデフォルトで calc_t=true にする。
        if (node.type == "aircon" && !hasCalcT) {
            node.calc_t = true;
            if (verbosity >= 1) {
                writeLog(logs, "  [INFO] aircon node: calc_t が未指定のため true をデフォルト適用しました: key=" + node.key);
            }
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

#include "branches_parser.h"
#include "parser_utils.h"
#include "utils/utils.h"
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>

using nlohmann::json;

namespace { }

std::vector<EdgeProperties> parseVentilationBranches(const json& config, std::ostream& logs, long timestep) {
    std::vector<EdgeProperties> branches;

    if (!config.contains("ventilation_branches") || !config["ventilation_branches"].is_array()) {
        writeLog(logs, "  [WARN] ventilation_branches 配列が見つかりません。");
        return branches;
    }

    const int verbosity = parser_utils::readVerbosity(config);
    const size_t total = config["ventilation_branches"].size();
    size_t index = 0;
    std::set<std::string> seenKeys;  // 既に見たブランチ名を記録

    for (const auto& branchJson : config["ventilation_branches"]) {
        ++index;
        EdgeProperties branch{};
        const std::string branchPrefix = "ventilation_branches[" + std::to_string(index-1) + "]";

        parser_utils::checkStringIfPresent(branchJson, "key", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "type", branchPrefix);
        if (branchJson.contains("key"))   branch.key   = branchJson["key"].get<std::string>();
        if (branchJson.contains("type"))  branch.type  = branchJson["type"].get<std::string>();

        // 重複チェック: 同じブランチ名が既に存在する場合はエラー
        if (!branch.key.empty()) {
            if (seenKeys.find(branch.key) != seenKeys.end()) {
                throw std::runtime_error("ventilation_branches: 重複するブランチ名が検出されました: \"" + branch.key + "\"");
            }
            seenKeys.insert(branch.key);
        }

        // unique_idはkeyと同じ値に設定
        branch.unique_id = branch.key;

        // 詳細
        parser_utils::checkStringIfPresent(branchJson, "subtype", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "comment", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "source", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "target", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "alpha", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "area", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "a", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "n", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "eta", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "p_max", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "p1", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "q_max", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "q1", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "h_from", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "h_to", branchPrefix);
        if (branchJson.contains("subtype")) branch.subtype = branchJson["subtype"].get<std::string>();
        if (branchJson.contains("comment")) branch.comment = branchJson["comment"].get<std::string>();
        if (branchJson.contains("source"))  branch.source  = branchJson["source"].get<std::string>();
        if (branchJson.contains("target"))  branch.target  = branchJson["target"].get<std::string>();
        if (branchJson.contains("alpha"))   branch.alpha   = branchJson["alpha"].get<double>();
        if (branchJson.contains("area"))    branch.area    = branchJson["area"].get<double>();
        if (branchJson.contains("a"))       branch.a       = branchJson["a"].get<double>();
        if (branchJson.contains("n"))       branch.n       = branchJson["n"].get<double>();
        if (branchJson.contains("eta"))     branch.eta     = branchJson["eta"].get<double>();
        if (branchJson.contains("p_max"))   branch.p_max   = branchJson["p_max"].get<double>();
        if (branchJson.contains("p1"))      branch.p1      = branchJson["p1"].get<double>();
        if (branchJson.contains("q_max"))   branch.q_max   = branchJson["q_max"].get<double>();
        if (branchJson.contains("q1"))      branch.q1      = branchJson["q1"].get<double>();
        if (branchJson.contains("h_from"))  branch.h_from  = branchJson["h_from"].get<double>();
        if (branchJson.contains("h_to"))    branch.h_to    = branchJson["h_to"].get<double>();

        // 時系列（配列/単一両対応）
        if (branchJson.contains("vol")) {
            branch.current_vol = parser_utils::readScalarOrSeries<double>(
                branchJson["vol"],
                branch.vol,
                static_cast<size_t>(timestep),
                0.0,
                branchPrefix + ".vol");
        }
        // enable が無い場合は「有効」を既定とする（旧入力との互換性・直感に合わせる）
        if (!branchJson.contains("enable")) {
            branch.current_enabled = true;
        }
        if (branchJson.contains("enable")) {
            const auto& ej = branchJson["enable"];
            if (ej.is_array()) {
                for (const auto& v : ej) {
                    if (!v.is_boolean()) {
                        throw std::runtime_error("ventilation_branches[" + std::to_string(index-1) + "].enable must be array<boolean>");
                    }
                }
                branch.enabled = ej.get<std::vector<bool>>();
                if (!branch.enabled.empty()) {
                    branch.current_enabled = static_cast<size_t>(timestep) < branch.enabled.size()
                        ? branch.enabled[static_cast<size_t>(timestep)]
                        : branch.enabled.back();
                }
            } else if (ej.is_boolean()) {
                branch.current_enabled = ej.get<bool>();
            } else {
                throw std::runtime_error("ventilation_branches[" + std::to_string(index-1) + "].enable must be boolean or array<boolean>");
            }
        }

        branches.push_back(std::move(branch));

        if (verbosity >= 2) {
            std::ostringstream oss;
            oss << "    換気ブランチ: " << branches.back().key << " (タイプ: " << branches.back().type << ") ("
                << index << "/" << total << ")";
            writeLog(logs, oss.str());
        }
    }

    {
        std::ostringstream oss;
        oss << "  全ての換気ブランチを読み込みました: " << branches.size() << "個";
        writeLog(logs, oss.str());
    }
    return branches;
}

std::vector<EdgeProperties> parseThermalBranches(const json& config, std::ostream& logs, long timestep) {
    std::vector<EdgeProperties> branches;

    if (!config.contains("thermal_branches") || !config["thermal_branches"].is_array()) {
        writeLog(logs, "  [WARN] thermal_branches 配列が見つかりません。");
        return branches;
    }

    const int verbosity = parser_utils::readVerbosity(config);
    const size_t total = config["thermal_branches"].size();
    size_t index = 0;
    std::set<std::string> seenKeys;  // 既に見たブランチ名を記録

    for (const auto& branchJson : config["thermal_branches"]) {
        ++index;
        EdgeProperties branch{};
        const std::string branchPrefix = "thermal_branches[" + std::to_string(index-1) + "]";

        if (branchJson.contains("key") && !branchJson["key"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].key must be string");
        if (branchJson.contains("type") && !branchJson["type"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].type must be string");
        if (branchJson.contains("key"))   branch.key   = branchJson["key"].get<std::string>();
        if (branchJson.contains("type"))  branch.type  = branchJson["type"].get<std::string>();

        // 重複チェック: 同じブランチ名が既に存在する場合はエラー
        if (!branch.key.empty()) {
            if (seenKeys.find(branch.key) != seenKeys.end()) {
                throw std::runtime_error("thermal_branches: 重複するブランチ名が検出されました: \"" + branch.key + "\"");
            }
            seenKeys.insert(branch.key);
        }

        // unique_idはkeyと同じ値に設定
        branch.unique_id = branch.key;

        // 詳細
        if (branchJson.contains("subtype") && !branchJson["subtype"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].subtype must be string");
        if (branchJson.contains("comment") && !branchJson["comment"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].comment must be string");
        if (branchJson.contains("source") && !branchJson["source"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].source must be string");
        if (branchJson.contains("target") && !branchJson["target"].is_string())
            throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].target must be string");
        parser_utils::checkNumberIfPresent(branchJson, "conductance", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "area", branchPrefix);
        if (branchJson.contains("subtype"))      branch.subtype      = branchJson["subtype"].get<std::string>();
        if (branchJson.contains("comment"))      branch.comment      = branchJson["comment"].get<std::string>();
        if (branchJson.contains("source"))       branch.source       = branchJson["source"].get<std::string>();
        if (branchJson.contains("target"))       branch.target       = branchJson["target"].get<std::string>();
        if (branchJson.contains("conductance"))  branch.conductance  = branchJson["conductance"].get<double>();
        if (branchJson.contains("area"))         branch.area         = branchJson["area"].get<double>();

        // response_conduction 用係数（配列）
        auto readNumberArray = [&](const json& obj, const char* key, std::vector<double>& out) {
            if (!obj.contains(key)) return false;
            const auto& v = obj[key];
            if (!v.is_array()) {
                throw std::runtime_error(branchPrefix + std::string(".") + key + " must be array<number>");
            }
            out.clear();
            out.reserve(v.size());
            for (size_t i2 = 0; i2 < v.size(); ++i2) {
                if (!v[i2].is_number()) {
                    throw std::runtime_error(branchPrefix + std::string(".") + key + " must be array<number>");
                }
                out.push_back(v[i2].get<double>());
            }
            return true;
        };

        if (branch.type == "response_conduction") {
            // area は必須（係数は q''[W/m2] のため solver 側で A を掛けて q[W] にする）
            if (!branchJson.contains("area") || !branchJson["area"].is_number()) {
                throw std::runtime_error(branchPrefix + " response_conduction requires numeric 'area' (per m2 coefficients)");
            }
            branch.area = branchJson["area"].get<double>();
            if (branch.area <= 0.0) {
                throw std::runtime_error(branchPrefix + " response_conduction requires positive 'area'");
            }

            // 必須: a/b は少なくとも現在係数（len>=1）
            const bool hasA1 = readNumberArray(branchJson, "resp_a_src", branch.resp_a_src);
            const bool hasB1 = readNumberArray(branchJson, "resp_b_src", branch.resp_b_src);
            const bool hasA2 = readNumberArray(branchJson, "resp_a_tgt", branch.resp_a_tgt);
            const bool hasB2 = readNumberArray(branchJson, "resp_b_tgt", branch.resp_b_tgt);
            // 任意
            (void)readNumberArray(branchJson, "resp_c_src", branch.resp_c_src);
            (void)readNumberArray(branchJson, "resp_c_tgt", branch.resp_c_tgt);

            if (!hasA1 || !hasB1 || !hasA2 || !hasB2 ||
                branch.resp_a_src.empty() || branch.resp_b_src.empty() ||
                branch.resp_a_tgt.empty() || branch.resp_b_tgt.empty()) {
                throw std::runtime_error(
                    branchPrefix + " response_conduction requires non-empty resp_a_src/resp_b_src/resp_a_tgt/resp_b_tgt");
            }

            if (branch.resp_a_src.size() != branch.resp_b_src.size()) {
                throw std::runtime_error(branchPrefix + " response_conduction requires resp_a_src and resp_b_src same length");
            }
            if (branch.resp_a_tgt.size() != branch.resp_b_tgt.size()) {
                throw std::runtime_error(branchPrefix + " response_conduction requires resp_a_tgt and resp_b_tgt same length");
            }
            // 推奨（builder自動生成系）: len(c)=len(a)-1。手入力の簡略系として空は許容。
            if (!branch.resp_c_src.empty() && branch.resp_c_src.size() != branch.resp_a_src.size() - 1) {
                throw std::runtime_error(branchPrefix + " response_conduction resp_c_src length should be len(resp_a_src)-1 or empty");
            }
            if (!branch.resp_c_tgt.empty() && branch.resp_c_tgt.size() != branch.resp_a_tgt.size() - 1) {
                throw std::runtime_error(branchPrefix + " response_conduction resp_c_tgt length should be len(resp_a_tgt)-1 or empty");
            }
        }

        // 時系列（配列/単一両対応）
        if (branchJson.contains("heat_generation")) {
            branch.current_heat_generation = parser_utils::readScalarOrSeries<double>(
                branchJson["heat_generation"],
                branch.heat_generation,
                static_cast<size_t>(timestep),
                0.0,
                branchPrefix + ".heat_generation");
        }
        if (branchJson.contains("enable")) {
            const auto& ej = branchJson["enable"];
            if (ej.is_array()) {
                for (const auto& v : ej) {
                    if (!v.is_boolean()) {
                        throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].enable must be array<boolean>");
                    }
                }
                branch.enabled = ej.get<std::vector<bool>>();
                if (!branch.enabled.empty()) {
                    branch.current_enabled = static_cast<size_t>(timestep) < branch.enabled.size()
                        ? branch.enabled[static_cast<size_t>(timestep)]
                        : branch.enabled.back();
                }
            } else if (ej.is_boolean()) {
                branch.current_enabled = ej.get<bool>();
            } else {
                throw std::runtime_error("thermal_branches[" + std::to_string(index-1) + "].enable must be boolean or array<boolean>");
            }
        } else {
            // enable が無い場合は既定で有効
            branch.current_enabled = true;
        }

        branches.push_back(std::move(branch));

        if (verbosity >= 2) {
            std::ostringstream oss;
            oss << "    熱ブランチ: " << branches.back().key << " (タイプ: " << branches.back().type << ") ("
                << index << "/" << total << ")";
            writeLog(logs, oss.str());
        }
    }

    {
        std::ostringstream oss;
        oss << "  全ての熱ブランチを読み込みました: " << branches.size() << "個";
        writeLog(logs, oss.str());
    }
    return branches;
}



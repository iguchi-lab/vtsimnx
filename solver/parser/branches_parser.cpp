#include "branches_parser.h"
#include "parser_utils.h"
#include "utils/utils.h"
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>

using nlohmann::json;

namespace {

static inline std::string requireStringAndGet(const json& obj,
                                              const char* key,
                                              const std::string& prefix) {
    parser_utils::requireString(obj, key, prefix);
    return obj[key].get<std::string>();
}

static inline void parseEnableField(const json& obj,
                                   std::vector<bool>& storage,
                                   bool& current,
                                   long timestep,
                                   const std::string& prefix) {
    const std::string path = parser_utils::makePath(prefix, "enable");
    if (!obj.contains("enable")) {
        current = true;
        return;
    }
    const auto& ej = obj["enable"];
    if (ej.is_array()) {
        storage.clear();
        storage.reserve(ej.size());
        for (size_t i = 0; i < ej.size(); ++i) {
            if (!ej[i].is_boolean()) {
                throw std::runtime_error(path + " must be boolean or array<boolean>");
            }
            storage.push_back(ej[i].get<bool>());
        }
        if (!storage.empty()) {
            const size_t ts = timestep < 0 ? 0u : static_cast<size_t>(timestep);
            current = parser_utils::valueOrLast(storage, ts, true);
        } else {
            // 空配列は「未指定扱い」と同等にする（既定=有効）
            current = true;
        }
        return;
    }
    if (ej.is_boolean()) {
        current = ej.get<bool>();
        return;
    }
    throw std::runtime_error(path + " must be boolean or array<boolean>");
}

static inline void ensureUniqueKeyOrThrow(const std::string& where,
                                         const std::string& key,
                                         std::set<std::string>& seen) {
    if (key.empty()) {
        throw std::runtime_error(where + ".key is required");
    }
    if (seen.find(key) != seen.end()) {
        throw std::runtime_error(where + ": 重複するブランチ名が検出されました: \"" + key + "\"");
    }
    seen.insert(key);
}

} // namespace

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

        branch.key  = requireStringAndGet(branchJson, "key", branchPrefix);
        branch.type = requireStringAndGet(branchJson, "type", branchPrefix);

        // 重複チェック: 同じブランチ名が既に存在する場合はエラー
        ensureUniqueKeyOrThrow("ventilation_branches", branch.key, seenKeys);

        // unique_idはkeyと同じ値に設定
        branch.unique_id = branch.key;

        // 詳細
        parser_utils::checkStringIfPresent(branchJson, "subtype", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "comment", branchPrefix);
        branch.source = requireStringAndGet(branchJson, "source", branchPrefix);
        branch.target = requireStringAndGet(branchJson, "target", branchPrefix);
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
        if (branchJson.contains("humidity_generation")) {
            branch.current_humidity_generation = parser_utils::readScalarOrSeries<double>(
                branchJson["humidity_generation"],
                branch.humidity_generation,
                static_cast<size_t>(timestep),
                0.0,
                branchPrefix + ".humidity_generation");
        }
        if (branchJson.contains("dust_generation")) {
            branch.current_dust_generation = parser_utils::readScalarOrSeries<double>(
                branchJson["dust_generation"],
                branch.dust_generation,
                static_cast<size_t>(timestep),
                0.0,
                branchPrefix + ".dust_generation");
        }
        parseEnableField(branchJson, branch.enabled, branch.current_enabled, timestep, branchPrefix);

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

        branch.key  = requireStringAndGet(branchJson, "key", branchPrefix);
        branch.type = requireStringAndGet(branchJson, "type", branchPrefix);

        // 重複チェック: 同じブランチ名が既に存在する場合はエラー
        ensureUniqueKeyOrThrow("thermal_branches", branch.key, seenKeys);

        // unique_idはkeyと同じ値に設定
        branch.unique_id = branch.key;

        // 詳細
        parser_utils::checkStringIfPresent(branchJson, "subtype", branchPrefix);
        parser_utils::checkStringIfPresent(branchJson, "comment", branchPrefix);
        branch.source = requireStringAndGet(branchJson, "source", branchPrefix);
        branch.target = requireStringAndGet(branchJson, "target", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "conductance", branchPrefix);
        parser_utils::checkNumberIfPresent(branchJson, "area", branchPrefix);
        if (branchJson.contains("subtype"))      branch.subtype      = branchJson["subtype"].get<std::string>();
        if (branchJson.contains("comment"))      branch.comment      = branchJson["comment"].get<std::string>();
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
        parseEnableField(branchJson, branch.enabled, branch.current_enabled, timestep, branchPrefix);

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



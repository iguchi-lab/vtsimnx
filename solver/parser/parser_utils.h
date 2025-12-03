#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace parser_utils {

    // 与えられたインデックスの要素を返す。範囲外なら末尾、空ならフォールバック
    template <typename T>
    inline const T& valueOrLast(const std::vector<T>& vec, size_t idx, const T& fallback) {
        if (!vec.empty()) {
            if (idx < vec.size()) return vec[idx];
            return vec.back();
        }
        return fallback;
    }

    // ログ冗長度を設定から読み取る（なければ既定値 1）
    inline int readVerbosity(const nlohmann::json& config) {
        try {
            if (config.contains("simulation") &&
                config["simulation"].contains("log") &&
                config["simulation"]["log"].contains("verbosity")) {
                return config["simulation"]["log"]["verbosity"].get<int>();
            }
        } catch (...) {}
        return 1;
    }

    // 真偽値を "true"/"false" に変換
    inline std::string boolToString(bool v) {
        return v ? "true" : "false";
    }

} // namespace parser_utils



#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>
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

    template <typename T>
    inline T readScalarOrSeries(const nlohmann::json& value,
                                std::vector<T>& storage,
                                size_t timestep,
                                const T& fallback,
                                const std::string& fieldPath) {
        static_assert(std::is_arithmetic_v<T>, "readScalarOrSeries requires arithmetic types");

        auto makeError = [&](const std::string& suffix) {
            throw std::runtime_error(fieldPath + suffix);
        };

        if (value.is_array()) {
            storage.clear();
            for (size_t idx = 0; idx < value.size(); ++idx) {
                const auto& element = value[idx];
                if (!element.is_number()) {
                    makeError(" must be array<number>");
                }
                storage.push_back(element.get<T>());
            }
            return valueOrLast(storage, timestep, fallback);
        }

        if (!value.is_number()) {
            makeError(" must be number or array<number>");
        }
        return value.get<T>();
    }

} // namespace parser_utils



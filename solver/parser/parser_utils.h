#pragma once

#include <string>
#include <string_view>
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

    // -----------------------------------------------------------------
    // JSON フィールド検証/取得（重複排除用）
    // - required: 無ければ例外
    // - optional: あれば型チェック
    // 例外メッセージは「{prefix}.{key} ...」形式に統一する
    // -----------------------------------------------------------------

    inline std::string makePath(const std::string& prefix, std::string_view key) {
        if (prefix.empty()) return std::string(key);
        return prefix + "." + std::string(key);
    }

    inline void requireString(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string path = makePath(prefix, key);
        if (!obj.contains(std::string(key))) {
            throw std::runtime_error(path + " is required");
        }
        if (!obj[std::string(key)].is_string()) {
            throw std::runtime_error(path + " must be string");
        }
    }

    inline void requireNumber(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string path = makePath(prefix, key);
        if (!obj.contains(std::string(key))) {
            throw std::runtime_error(path + " is required");
        }
        if (!obj[std::string(key)].is_number()) {
            throw std::runtime_error(path + " must be number");
        }
    }

    inline void requireBoolean(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string path = makePath(prefix, key);
        if (!obj.contains(std::string(key))) {
            throw std::runtime_error(path + " is required");
        }
        if (!obj[std::string(key)].is_boolean()) {
            throw std::runtime_error(path + " must be boolean");
        }
    }

    inline void requireArray(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string path = makePath(prefix, key);
        if (!obj.contains(std::string(key))) {
            throw std::runtime_error(path + " is required");
        }
        if (!obj[std::string(key)].is_array()) {
            throw std::runtime_error(path + " must be array");
        }
    }

    inline void checkStringIfPresent(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string k(key);
        if (!obj.contains(k)) return;
        const std::string path = makePath(prefix, key);
        if (!obj[k].is_string()) {
            throw std::runtime_error(path + " must be string");
        }
    }

    inline void checkNumberIfPresent(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string k(key);
        if (!obj.contains(k)) return;
        const std::string path = makePath(prefix, key);
        if (!obj[k].is_number()) {
            throw std::runtime_error(path + " must be number");
        }
    }

    inline void checkBooleanIfPresent(const nlohmann::json& obj, std::string_view key, const std::string& prefix) {
        const std::string k(key);
        if (!obj.contains(k)) return;
        const std::string path = makePath(prefix, key);
        if (!obj[k].is_boolean()) {
            throw std::runtime_error(path + " must be boolean");
        }
    }

    inline std::string getStringIfPresent(const nlohmann::json& obj,
                                          std::string_view key,
                                          const std::string& prefix,
                                          const std::string& fallback = {}) {
        checkStringIfPresent(obj, key, prefix);
        const std::string k(key);
        if (!obj.contains(k)) return fallback;
        return obj[k].get<std::string>();
    }

    inline double getNumberIfPresent(const nlohmann::json& obj,
                                     std::string_view key,
                                     const std::string& prefix,
                                     double fallback) {
        checkNumberIfPresent(obj, key, prefix);
        const std::string k(key);
        if (!obj.contains(k)) return fallback;
        return obj[k].get<double>();
    }

    inline bool getBooleanIfPresent(const nlohmann::json& obj,
                                    std::string_view key,
                                    const std::string& prefix,
                                    bool fallback) {
        checkBooleanIfPresent(obj, key, prefix);
        const std::string k(key);
        if (!obj.contains(k)) return fallback;
        return obj[k].get<bool>();
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



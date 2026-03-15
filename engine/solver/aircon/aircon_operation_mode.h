#pragma once

#include <algorithm>
#include <cctype>
#include <string>

enum class OperationMode {
    Heating,
    Cooling,
};

inline std::string toLowerCopy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

inline bool isHeating(OperationMode mode) {
    return mode == OperationMode::Heating;
}

inline const char* modeKey(OperationMode mode) {
    return isHeating(mode) ? "heating" : "cooling";
}

inline OperationMode resolveOperationModeForRuntime(const std::string& requestedMode,
                                                    double indoorTemp,
                                                    double airconTemp) {
    if (requestedMode == "HEATING") return OperationMode::Heating;
    if (requestedMode == "COOLING") return OperationMode::Cooling;
    if (requestedMode == "AUTO") {
        return (indoorTemp > airconTemp) ? OperationMode::Cooling : OperationMode::Heating;
    }
    // 既存挙動互換: 不明モードは cooling 扱い
    return OperationMode::Cooling;
}

inline OperationMode parseOperationModeOrDefaultCooling(const std::string& mode) {
    return toLowerCopy(mode) == "heating" ? OperationMode::Heating : OperationMode::Cooling;
}

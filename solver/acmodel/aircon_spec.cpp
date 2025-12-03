#include "aircon_spec.h"
#include "aircon_constants.h"
#include <algorithm>

namespace acmodel {

AirconSpec::AirconSpec(const nlohmann::json& spec) : spec_(spec) {
    // 基本仕様パラメータの初期化
    rated_capacity = spec.value("rated_capacity", 2400.0);  // デフォルト2400W
    cop = spec.value("cop", 3.0);  // デフォルトCOP 3.0
    max_capacity = spec.value("max_capacity", rated_capacity * 1.2);  // デフォルト定格の1.2倍
    model_name = spec.value("model_name", "unknown");
    
    calculateCOP();
}

bool AirconSpec::validateSpec() const {
    // 基本的なバリデーション
    for (const auto& mode : Constants::MODES) {
        if (!spec_.contains("Q") || !spec_["Q"].contains(mode)) {
            return false;
        }
        if (!spec_.contains("P") || !spec_["P"].contains(mode)) {
            return false;
        }
    }
    return true;
}

void AirconSpec::calculateCOP() {
    if (!spec_.contains("Q") || !spec_.contains("P")) {
        return;
    }

    for (const auto& mode : Constants::MODES) {
        if (spec_["Q"].contains(mode) && spec_["P"].contains(mode)) {
            const auto& Q_mode = spec_["Q"][mode];
            const auto& P_mode = spec_["P"][mode];
            
            for (auto it = Q_mode.begin(); it != Q_mode.end(); ++it) {
                const std::string& key = it.key();
                if (P_mode.contains(key)) {
                    double Q_val = it.value().get<double>();
                    double P_val = P_mode[key].get<double>();
                    
                    if (P_val > 0) {
                        COP_[mode][key] = Q_val / P_val;
                    }
                }
            }
        }
    }
}

double AirconSpec::getCOP(const std::string& mode, const std::string& key) const {
    auto mode_it = COP_.find(mode);
    if (mode_it != COP_.end()) {
        auto key_it = mode_it->second.find(key);
        if (key_it != mode_it->second.end()) {
            return key_it->second;
        }
    }
    return 0.0;
}

double AirconSpec::getCapacity(const std::string& mode, const std::string& key) const {
    if (spec_.contains("Q") && spec_["Q"].contains(mode) && spec_["Q"][mode].contains(key)) {
        return spec_["Q"][mode][key].get<double>();
    }
    return 0.0;
}

double AirconSpec::getPower(const std::string& mode, const std::string& key) const {
    if (spec_.contains("P") && spec_["P"].contains(mode) && spec_["P"][mode].contains(key)) {
        return spec_["P"][mode][key].get<double>();
    }
    return 0.0;
}

double AirconSpec::getVolume(const std::string& volumeType, const std::string& mode, const std::string& key) const {
    if (spec_.contains(volumeType) && spec_[volumeType].contains(mode) && spec_[volumeType][mode].contains(key)) {
        return spec_[volumeType][mode][key].get<double>();
    }
    return 0.0;
}

double AirconSpec::getFanPower(const std::string& mode, const std::string& key) const {
    if (spec_.contains("P_fan") && spec_["P_fan"].contains(mode) && spec_["P_fan"][mode].contains(key)) {
        return spec_["P_fan"][mode][key].get<double>();
    }
    return 0.0;
}

} // namespace acmodel 
#pragma once

#include <cmath>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

// エアコンのモード別性能値
struct AirconModeSpec {
    std::optional<double> min;
    std::optional<double> mid;
    std::optional<double> rtd;
    std::optional<double> max;
    std::optional<double> dsgn;

    void setFromJson(const nlohmann::json& json) {
        if (json.contains("min")) min = json["min"].get<double>();
        if (json.contains("mid")) mid = json["mid"].get<double>();
        if (json.contains("rtd")) rtd = json["rtd"].get<double>();
        if (json.contains("max")) max = json["max"].get<double>();
        if (json.contains("dsgn")) dsgn = json["dsgn"].get<double>();
    }

    std::optional<double> getValue(const std::string& rating = "rtd") const {
        if (rating == "min" && min.has_value()) return min;
        if (rating == "mid" && mid.has_value()) return mid;
        if (rating == "rtd" && rtd.has_value()) return rtd;
        if (rating == "max" && max.has_value()) return max;
        if (rating == "dsgn" && dsgn.has_value()) return dsgn;
        return std::nullopt;
    }
};

// エアコンの性能項目
struct AirconPerformanceSpec {
    AirconModeSpec cooling;
    AirconModeSpec heating;

    void setFromJson(const nlohmann::json& json) {
        if (json.contains("cooling")) cooling.setFromJson(json["cooling"]);
        if (json.contains("heating")) heating.setFromJson(json["heating"]);
    }

    std::optional<double> getValue(const std::string& mode, const std::string& rating = "rtd") const {
        if (mode == "cooling") return cooling.getValue(rating);
        if (mode == "heating") return heating.getValue(rating);
        return std::nullopt;
    }
};

// エアコンの仕様
struct AirconSpec {
    AirconPerformanceSpec Q;   // 能力 [kW]（カタログ。COP 用 Q.min を含む）
    AirconPerformanceSpec P;   // 消費電力 [kW]
    std::optional<AirconPerformanceSpec> P_fan;   // ファン消費電力 [kW]
    std::optional<AirconPerformanceSpec> V_inner; // 内部風量 [m³/s]
    std::optional<AirconPerformanceSpec> V_outer; // 外部風量 [m³/s]

    // 制御用の最低処理能力 [kW]（任意）。カタログ Q.min とは独立。
    // 指定時、設定維持中の |必要負荷| がこの値以下なら最低能力分を処理する。
    std::optional<double> min_process_heating;
    std::optional<double> min_process_cooling;

    // 追加パラメータ
    double max_heat_capacity = 2400.0; // 最大処理熱量 [W]

    static AirconSpec fromJson(const nlohmann::json& ac_spec_json) {
        AirconSpec spec;

        if (ac_spec_json.contains("Q")) spec.Q.setFromJson(ac_spec_json["Q"]);
        if (ac_spec_json.contains("P")) spec.P.setFromJson(ac_spec_json["P"]);

        if (ac_spec_json.contains("P_fan")) {
            spec.P_fan = AirconPerformanceSpec();
            spec.P_fan->setFromJson(ac_spec_json["P_fan"]);
        }

        if (ac_spec_json.contains("V_inner")) {
            spec.V_inner = AirconPerformanceSpec();
            spec.V_inner->setFromJson(ac_spec_json["V_inner"]);
        }

        if (ac_spec_json.contains("V_outer")) {
            spec.V_outer = AirconPerformanceSpec();
            spec.V_outer->setFromJson(ac_spec_json["V_outer"]);
        }

        if (ac_spec_json.contains("min_process") && ac_spec_json["min_process"].is_object()) {
            const auto& mp = ac_spec_json["min_process"];
            if (mp.contains("heating") && mp["heating"].is_number()) {
                const double v = mp["heating"].get<double>();
                if (std::isfinite(v) && v > 0.0) spec.min_process_heating = v;
            }
            if (mp.contains("cooling") && mp["cooling"].is_number()) {
                const double v = mp["cooling"].get<double>();
                if (std::isfinite(v) && v > 0.0) spec.min_process_cooling = v;
            }
        }

        if (ac_spec_json.contains("max_heat_capacity")) {
            spec.max_heat_capacity = ac_spec_json["max_heat_capacity"].get<double>();
        }

        return spec;
    }

    double getMaxHeatCapacity() const { return max_heat_capacity; }
    std::optional<double> getCapacity(const std::string& mode, const std::string& rating = "rtd") const {
        return Q.getValue(mode, rating);
    }
    /** 制御用最低処理能力 [kW]。未指定なら nullopt（カタログ Q.min は見ない）。 */
    std::optional<double> getMinProcess(const std::string& mode) const {
        if (mode == "cooling") return min_process_cooling;
        if (mode == "heating") return min_process_heating;
        return std::nullopt;
    }
    /** 能力上限 [kW]。max があればそれ、なければ mid を返す（DUCT_CENTRAL / LATENT_EVALUATE で mid のみの spec に対応） */
    std::optional<double> getCapacityMaxForMode(const std::string& mode) const {
        auto v = Q.getValue(mode, "max");
        if (v.has_value() && *v > 0) return v;
        return Q.getValue(mode, "mid");
    }
    std::optional<double> getPower(const std::string& mode, const std::string& rating = "rtd") const {
        return P.getValue(mode, rating);
    }
    std::optional<double> getInnerFlow(const std::string& mode, const std::string& rating = "rtd") const {
        if (V_inner.has_value()) return V_inner->getValue(mode, rating);
        return std::nullopt;
    }
};



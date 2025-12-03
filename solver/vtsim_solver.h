#pragma once

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <limits>
#include <optional>

// === 物理定数 ===
namespace PhysicalConstants {
    const double DENSITY_DRY_AIR = 1.2;             // 標準空気密度 [kg/m³]
    const double SPECIFIC_HEAT_AIR = 1.005;         // 空気の定圧比熱 [kJ/(kg·K)]
    const double LATENT_HEAT_VAPORIZATION = 2450.0; // 蒸発潜熱 [kJ/kg]
    const double DEFAULT_VENTILATION_RATE = 28.2;   // デフォルト換気量 [m³/min]
}

// === 許容誤差定数 ===
namespace ToleranceConstants {
    const double TEMPERATURE_TOLERANCE = 1e-3;      // 温度許容誤差 [K]
    const double CAPACITY_TOLERANCE = 1e-6;         // 能力許容誤差
    const double BOUNDS_TOLERANCE = 1e-6;           // 境界値許容誤差 [K]
    const double DEFAULT_HUMIDITY = 0.0;            // デフォルト湿度
    const double DEFAULT_LATENT_HEAT = 0.0;         // デフォルト潜熱
}

// 型エイリアス（プロジェクト全体で使用）
using PressureMap    = std::map<std::string, double>;                          // 圧力マップ {node} -> [Pa]
using FlowRateMap    = std::map<std::pair<std::string, std::string>, double>;  // 風量マップ {source, target} -> [kg/s]
using FlowBalanceMap = std::map<std::string, double>;                           // 流量バランスマップ {node} -> [kg/s]

using TemperatureMap = std::map<std::string, double>;                           // 温度マップ {node} -> [K]
using HeatRateMap    = std::map<std::pair<std::string, std::string>, double>;  // 熱流量マップ {source, target} -> [W]
using HeatBalanceMap = std::map<std::string, double>;                           // 熱バランスマップ {node} -> [W]

using AirconDataMap  = std::map<std::string, double>;  // エアコンデータマップ {node} -> [値]

// シミュレーション定数を格納する構造体
struct SimulationConstants {
    std::string startTime;
    std::string endTime;
    int timestep;
    int length;
    double ventilationTolerance;
    double thermalTolerance;
    double convergenceTolerance;
    double maxInnerIteration;
    bool pressureCalc;
    bool temperatureCalc;
    bool humidityCalc = false;
    bool concentrationCalc = false;
    // 追加: ログ・出力制御
    // 0: silent, 1: normal, 2: verbose, 3: debug
    int logVerbosity = 1;
    // フォールバック詳細ログを有効化
    bool logFallbackDetails = true;
    // フォールバック関連の補助出力(csv/txt)を有効化
    bool exportFallbackArtifacts = true;
};

// エアコンのモード別性能値
struct AirconModeSpec {
    std::optional<double> min;
    std::optional<double> mid;
    std::optional<double> rtd;
    std::optional<double> max;
    std::optional<double> dsgn;
    
    // JSONから値を設定
    void setFromJson(const nlohmann::json& json) {
        if (json.contains("min")) min = json["min"].get<double>();
        if (json.contains("mid")) mid = json["mid"].get<double>();
        if (json.contains("rtd")) rtd = json["rtd"].get<double>();
        if (json.contains("max")) max = json["max"].get<double>();
        if (json.contains("dsgn")) dsgn = json["dsgn"].get<double>();
    }
    
    // 指定されたrating値を取得
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
    
    // JSONから値を設定
    void setFromJson(const nlohmann::json& json) {
        if (json.contains("cooling")) cooling.setFromJson(json["cooling"]);
        if (json.contains("heating")) heating.setFromJson(json["heating"]);
    }
    
    // 指定されたモードとrating値を取得
    std::optional<double> getValue(const std::string& mode, const std::string& rating = "rtd") const {
        if (mode == "cooling") return cooling.getValue(rating);
        if (mode == "heating") return heating.getValue(rating);
        return std::nullopt;
    }
};

// エアコンの仕様
struct AirconSpec {
    AirconPerformanceSpec Q;      // 能力 [kW]
    AirconPerformanceSpec P;      // 消費電力 [kW]
    std::optional<AirconPerformanceSpec> P_fan;    // ファン消費電力 [kW]
    std::optional<AirconPerformanceSpec> V_inner;  // 内部風量 [m³/s]
    std::optional<AirconPerformanceSpec> V_outer;  // 外部風量 [m³/s]
    
    // 追加パラメータ
    double max_heat_capacity = 2400.0; // 最大処理熱量 [W]
    
    // JSONから構造体に変換
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
        
        if (ac_spec_json.contains("max_heat_capacity")) {
            spec.max_heat_capacity = ac_spec_json["max_heat_capacity"].get<double>();
        }
        
        return spec;
    }
    
    // 便利なアクセサメソッド
    double getMaxHeatCapacity() const { return max_heat_capacity; }
    std::optional<double> getCapacity(const std::string& mode, const std::string& rating = "rtd") const {
        return Q.getValue(mode, rating);
    }
    std::optional<double> getPower(const std::string& mode, const std::string& rating = "rtd") const {
        return P.getValue(mode, rating);
    }
    std::optional<double> getInnerFlow(const std::string& mode, const std::string& rating = "rtd") const {
        if (V_inner.has_value()) return V_inner->getValue(mode, rating);
        return std::nullopt;
    }
};

// ノード（頂点）のプロパティ
struct VertexProperties {
    std::string key;
    std::string name;
    std::string type;
    std::string subtype;
    std::string comment;
    std::string ref_node;
    std::string in_node;
    std::string set_node;
    std::string outside_node;
    std::string model;
    std::vector<std::string> mode;
    std::string current_mode;
    nlohmann::json ac_spec;
    bool calc_p = false;
    bool calc_t = false;
    bool calc_x = false;
    bool calc_c = false;
    std::vector<double> p;
    double current_p;
    std::vector<double> t;
    double current_t;
    std::vector<double> x;
    double current_x;
    std::vector<double> c;
    double current_c;
    double heat_source = 0.0;
    std::vector<double> pre_temp;
    double current_pre_temp = 20.0;
    double v = 0.0;
    bool on = false;
    std::optional<AirconSpec> aircon_spec;  // 新しい構造体
    
    // 時系列データ更新メソッド
    void updateForTimestep(int timestep) {
        if (!p.empty() && static_cast<size_t>(timestep) < p.size()) {
            current_p = p[timestep];
        }
        if (!t.empty() && static_cast<size_t>(timestep) < t.size()) {
            current_t = t[timestep];
        }
        if (!x.empty() && static_cast<size_t>(timestep) < x.size()) {
            current_x = x[timestep];
        }
        if (!c.empty() && static_cast<size_t>(timestep) < c.size()) {
            current_c = c[timestep];
        }
        if (!pre_temp.empty() && static_cast<size_t>(timestep) < pre_temp.size()) {
            current_pre_temp = pre_temp[timestep];
        }
        if (!mode.empty() && static_cast<size_t>(timestep) < mode.size()) {
            current_mode = mode[timestep];
        }
    }
    
    // アクセサメソッド
    const AirconSpec* getAirconSpec() const {
        if (aircon_spec.has_value()) {
            return &aircon_spec.value();
        }
        return nullptr;
    }
    
    // エアコン仕様の初期化
    void initializeAirconSpec() {
        if (type == "aircon" && !ac_spec.empty()) {
            aircon_spec = AirconSpec::fromJson(ac_spec);
        }
    }
};

// エッジ（ブランチ）のプロパティ
struct EdgeProperties {
    std::string key;
    std::string unique_id;  // 個別ブランチを識別するためのユニークID
    std::string type;
    std::string subtype;
    std::string comment;
    std::string source;
    std::string target;
    std::vector<bool> enabled;
    bool current_enabled;
    double alpha = 0.0;
    double area = 0.0;
    double a = 0.0;
    double n = 0.0;
    std::vector<double> vol;
    double current_vol;
    double h_from = 0.0;
    double h_to = 0.0;
    double p_max = 0.0;
    double p1 = 0.0;
    double q_max = 0.0;
    double q1 = 0.0;
    double eta = 0.0;
    double flow_rate = 0.0;
    double heat_rate = 0.0;  // 熱流量
    double conductance = 0.0;
    std::vector<double> heat_generation;
    double current_heat_generation;
    
    // 時系列データ更新メソッド
    void updateForTimestep(int timestep) {
        if (!enabled.empty() && static_cast<size_t>(timestep) < enabled.size()) {
            current_enabled = enabled[timestep];
        }
        if (!vol.empty() && static_cast<size_t>(timestep) < vol.size()) {
            current_vol = vol[timestep];
        }
        if (!heat_generation.empty() && static_cast<size_t>(timestep) < heat_generation.size()) {
            current_heat_generation = heat_generation[timestep];
        }
    }
};

// Boost Graphの定義
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties, EdgeProperties> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;

// 1タイムステップ分の結果を格納する構造体
struct TimestepResult {
    // 圧力関連（pressureCalcがtrueの場合のみ有効）
    PressureMap pressureMap;
    std::map<std::string, double> flowRateMap;  // 個別ブランチの風量データ
    FlowBalanceMap flowBalanceMap;

    // 温度関連（temperatureCalcがtrueの場合のみ有効）
    TemperatureMap temperatureMap;
    std::map<std::string, double> heatRateMap;  // 個別ブランチの熱流量データ
    HeatBalanceMap heatBalanceMap;
    
    // エアコン関連（temperatureCalcがtrueの場合のみ有効）
    AirconDataMap airconInletTempMap;
    AirconDataMap airconOutletTempMap;
    AirconDataMap airconFlowMap;
    AirconDataMap airconSensibleHeatMap;
    AirconDataMap airconLatentHeatMap;
    AirconDataMap airconPowerMap;
    AirconDataMap airconCOPMap;
    
    // その他（将来の拡張用）
    AirconDataMap humidityMap;
    AirconDataMap concentrationMap;
};

// シミュレーション結果の履歴を格納する構造体
struct SimulationResults {
    std::vector<TimestepResult> timestepHistory;  // タイムステップごとの結果履歴
};



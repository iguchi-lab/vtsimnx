#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <nlohmann/json.hpp>

#include "types/aircon_types.h"

// ノード（頂点）のプロパティ
struct VertexProperties {
    enum class TypeCode : std::uint8_t {
        Unknown = 0,
        Normal,
        Aircon,
        Capacity,
        Layer,
    };

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
    std::optional<AirconSpec> aircon_spec;

    // type（文字列）の比較をホットパスから外すためのキャッシュ
    mutable TypeCode type_code = TypeCode::Unknown;

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

    TypeCode getTypeCode() const {
        if (type_code != TypeCode::Unknown) return type_code;
        if (type == "aircon") type_code = TypeCode::Aircon;
        else if (type == "capacity") type_code = TypeCode::Capacity;
        else if (type == "layer") type_code = TypeCode::Layer;
        else if (type == "normal") type_code = TypeCode::Normal;
        else type_code = TypeCode::Unknown;
        return type_code;
    }

    const AirconSpec* getAirconSpec() const {
        if (aircon_spec.has_value()) {
            return &aircon_spec.value();
        }
        return nullptr;
    }

    void initializeAirconSpec() {
        if (type == "aircon" && !ac_spec.empty()) {
            aircon_spec = AirconSpec::fromJson(ac_spec);
        }
    }
};

// エッジ（ブランチ）のプロパティ
struct EdgeProperties {
    enum class TypeCode : std::uint8_t {
        Unknown = 0,
        Advection,
        Conductance,
        HeatGeneration,
    };

    std::string key;
    std::string unique_id; // 個別ブランチを識別するためのユニークID
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
    double heat_rate = 0.0; // 熱流量
    double conductance = 0.0;
    std::vector<double> heat_generation;
    double current_heat_generation;

    // type（文字列）の比較をホットパスから外すためのキャッシュ
    mutable TypeCode type_code = TypeCode::Unknown;

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

    TypeCode getTypeCode() const {
        if (type_code != TypeCode::Unknown) return type_code;
        if (type == "advection") type_code = TypeCode::Advection;
        else if (type == "conductance") type_code = TypeCode::Conductance;
        else if (type == "heat_generation") type_code = TypeCode::HeatGeneration;
        else type_code = TypeCode::Unknown;
        return type_code;
    }
};

// Boost Graphの定義
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties, EdgeProperties>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;



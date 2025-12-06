#pragma once

#include "../vtsim_solver.h"
#include <ceres/ceres.h>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

// 前方宣言
class ThermalNetwork;

// =============================================================================
// HeatBalanceConstraint - 熱バランス制約（Ceres用コスト関数）
// =============================================================================
class HeatBalanceConstraint {
public:
    HeatBalanceConstraint(
        const std::string& nodeName,
        const Graph& graph,
        const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
        const std::map<Vertex, size_t>& vertexToParameterIndex,
        const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges,
        const std::unordered_map<std::string, std::vector<Vertex>>& airconBySetNode,
        std::ostream& logFile
    ) : nodeName_(nodeName),
        graph_(graph),
        nodeKeyToVertex_(nodeKeyToVertex),
        vertexToParameterIndex_(vertexToParameterIndex),
        incidentEdges_(incidentEdges),
        airconBySetNode_(airconBySetNode),
        logFile_(logFile) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residual) const;

private:
    std::string nodeName_;
    const Graph& graph_;
    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
    const std::map<Vertex, size_t>& vertexToParameterIndex_;
    const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges_;
    const std::unordered_map<std::string, std::vector<Vertex>>& airconBySetNode_;
    std::ostream& logFile_;

    template <typename T>
    T getNodeTemperature(Vertex v, T const* const* parameters) const;

    const std::vector<Edge>& getIncidentEdges(Vertex v) const;
};

// =============================================================================
// ThermalSolver - 温度・熱流量計算のメインソルバー
// =============================================================================
class ThermalSolver {
public:
    ThermalSolver(ThermalNetwork& network, std::ostream& logFile);

    std::tuple<TemperatureMap, HeatRateMap, HeatBalanceMap> solveTemperatures(
        const SimulationConstants& constants);

    // システム診断情報の出力
    void outputThermalSystemDiagnostics();

private:
    ThermalNetwork& network_;
    std::ostream& logFile_;

    void setInitialTemperatures(std::vector<double>& temperatures, const std::vector<std::string>& nodeNames);
    
    TemperatureMap extractTemperatures(const std::vector<double>& temperatures, const std::vector<std::string>& nodeNames);
    
    HeatRateMap calculateHeatRates(const TemperatureMap& tempMap);
    
    HeatBalanceMap verifyBalance(const HeatRateMap& heatRates);
}; 
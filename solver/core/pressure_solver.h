#pragma once

#include "../vtsim_solver.h"
#include <ceres/ceres.h>
#include <functional>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>

// 前方宣言
class VentilationNetwork;

// 密度計算のヘルパー関数
double calculateDensity(double temperature);

#include "core/pressure_constraints.h"

// 圧力ソルバクラス
class PressureSolver {
public:
    using SolverResult = std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>;

    PressureSolver(VentilationNetwork& network, std::ostream& logFile);
    
    // Ceres Solver + LM法を使った圧力計算
    SolverResult solvePressures(const SimulationConstants& constants);
    
private:
    VentilationNetwork& network_;
    std::ostream& logFile_;

    struct SolverSetup {
        std::vector<std::string> nodeNames;
        std::vector<double> pressures;
        std::map<Vertex, size_t> vertexToParameterIndex;
    };
    struct StageAMapping {
        std::map<int, size_t> groupToParamIndex;
        std::map<Vertex, size_t> vertexToParamIndex;
        std::vector<std::string> nodeNames;
        size_t parameterCount = 0;
    };
    struct StageBSetup {
        std::map<Vertex, size_t> vertexToParamIndex;
        std::vector<std::string> nodeNames;
        std::vector<double> pressures;
    };
    
    // 初期圧力の設定
    void setInitialPressures(std::vector<double>& pressures, 
                           const std::vector<std::string>& nodeNames);
    bool initializeSolverSetup(SolverSetup& setup);
    
    // 圧力計算のヘルパー関数
    double calculateTotalPressure(double pressure, double temperature, double height) const;
    double calculatePressureDifference(
        const VertexProperties& sourceNode,
        const VertexProperties& targetNode,
        const EdgeProperties& edgeData,
        const PressureMap& pressureMap) const;
    void addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem);
    void runPrimarySolvers(const SimulationConstants& constants,
                           ceres::Problem& problem,
                           ceres::Solver::Summary& summary);
    bool runSolverTrial(const std::string& startLog,
                        const std::string& successLog,
                        ceres::Problem& problem,
                        ceres::Solver::Summary& summary,
                        double successTolerance,
                        const std::function<void(ceres::Solver::Options&)>& configureOptions);
    void restoreFixedFlowEdges(Graph& graph,
                               std::vector<std::string>& changedEdgeIds,
                               const std::map<std::string, std::string>& interfaceOriginalTypeById);
    std::map<std::string, std::string> captureInterfaceOriginalTypes(
        Graph& graph,
        const std::map<Vertex, int>& vertexToIndex,
        const std::vector<int>& groupOfVertex);
    StageAMapping buildStageAMapping(
        const Graph& graph,
        const std::vector<Vertex>& vertices,
        const std::vector<int>& groupOfVertex);
    std::vector<double> initializeStageAPressures(
        const Graph& graph,
        const StageAMapping& mapping,
        const PressureMap& prevPressureMapFB);
    void setupStageAProblem(
        ceres::Problem& problemFB,
        const StageAMapping& mapping,
        Graph& graph,
        const std::vector<Vertex>& vertices,
        const std::vector<int>& groupOfVertex,
        const PressureMap& prevPressureMapFB,
        std::vector<double>& pressuresFB,
        int superCountA);
    StageBSetup buildStageBSetup(
        const Graph& graph,
        const PressureMap& stageAPressureMap);
    bool runStageBTrials(const SimulationConstants& constants,
                         ceres::Problem& problemFB2,
                         ceres::Solver::Summary& fbSummary2,
                         const std::function<void(int, const std::string&)>& fallbackLog);
    std::optional<SolverResult> runFallbackLoop(
        const SimulationConstants& constants,
        SolverSetup& setup,
        ceres::Solver::Summary& summary);
    
    // 結果の取得
    PressureMap extractPressures(const std::vector<double>& pressures,
                                const std::vector<std::string>& nodeNames);
    
    // 流量計算
    FlowRateMap calculateFlowRates(const PressureMap& pressureMap);
    std::map<std::string, double> calculateIndividualFlowRates(const PressureMap& pressureMap);
    
    FlowBalanceMap verifyBalance(const FlowRateMap& flowRates);
    
    // 共通の流量計算
    double calculateFlowForEdge(const PressureMap& pressureMap, Edge edge);
};
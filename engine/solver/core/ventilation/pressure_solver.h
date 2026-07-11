#pragma once

#include "../../vtsim_solver.h"
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

    // 内部データは header から外に出して依存を軽量化（定義は .cpp 側）
    struct TrialResult;
    struct SolverSetup;
    struct StageAMapping;
    struct StageBSetup;
    
    // 初期圧力の設定
    void setInitialPressures(std::vector<double>& pressures, 
                           const std::vector<std::string>& nodeNames);
    bool initializeSolverSetup(SolverSetup& setup);
    
    // 圧力計算のヘルパー関数
    double calculateTotalPressure(double pressure, double temperature, double height) const;
    // pressure map に欠落がある場合は nullopt（0 Pa と区別する）
    std::optional<double> calculatePressureDifference(
        const VertexProperties& sourceNode,
        const VertexProperties& targetNode,
        const EdgeProperties& edgeData,
        const PressureMap& pressureMap) const;
    void addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem);
    void runPrimarySolvers(const SimulationConstants& constants,
                           ceres::Problem& problem,
                           ceres::Solver::Summary& summary);
    TrialResult runSolverTrial(const std::string& startLog,
                               const std::string& successLog,
                               ceres::Problem& problem,
                               ceres::Solver::Summary& summary,
                               double successTolerance,
                               const std::function<void(ceres::Solver::Options&)>& configureOptions,
                               std::function<void(const std::string&)> logger = {});
    TrialResult runTwoStageRelaxation(
        const SimulationConstants& constants,
        ceres::Problem& problem,
        ceres::Solver::Summary& summary,
        const std::string& labelStage1,
        const std::string& labelStage2,
        const std::function<void(const ceres::Solver::Summary&)>& afterStage1,
        std::function<void(const std::string&)> logger = {});
    TrialResult runUltraPreciseTrial(
        const SimulationConstants& constants,
        ceres::Problem& problem,
        ceres::Solver::Summary& summary,
        const std::string& labelTiming,
        double referenceCost,
        const std::function<void(double)>& onTolerance,
        std::function<void(const std::string&)> logger = {});
    void logCeresTiming(const std::string& label,
                        const ceres::Solver::Summary& summary,
                        std::function<void(const std::string&)> logger = {});
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
        int superCountA,
        const std::vector<std::vector<Edge>>& incidentEdgesByVertex);
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
    
    struct FlowComputationResult {
        FlowRateMap flows;
        bool ok = true;
        std::string detail;
    };

    // 流量計算（欠落圧力・非有限値は ok=false）
    FlowComputationResult calculateFlowRates(const PressureMap& pressureMap);
    std::optional<std::map<std::string, double>> calculateIndividualFlowRates(const PressureMap& pressureMap);
    
    FlowBalanceMap verifyBalance(const FlowRateMap& flowRates);
    
    // 共通の流量計算（失敗時は nullopt）
    std::optional<double> calculateFlowForEdge(const PressureMap& pressureMap, Edge edge) const;
};
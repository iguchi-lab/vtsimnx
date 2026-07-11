#pragma once

// PressureSolver 実装詳細（Ceres 依存）。ventilation 実装 .cpp からのみ include する。

#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/edge_mutation_guard.h"
#include "core/ventilation/pressure_balance.h"

#include <ceres/ceres.h>
#include <functional>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>

class PressureFallbackSolver;

struct PressureSolver::Impl {
    using SolverResult = PressureSolveResult; // 内部は詳細結果を返す

    Impl(VentilationNetwork& network, std::ostream& logFile);

    VentilationNetwork& network_;
    std::ostream& logFile_;

    struct TrialResult;
    struct SolverSetup;
    struct StageAMapping;
    struct StageBSetup;
    struct SupernodePartition;
    struct StageASolveResult;
    struct InterfaceFreezeResult;
    struct StageBSolveResult;
    struct FallbackOuterState;

    enum class FallbackOuterAction {
        AcceptSolution,
        ContinueOuter,
        StopOuter
    };

    using FallbackLogger = std::function<void(int, const std::string&)>;

    friend class PressureFallbackSolver;

    void setInitialPressures(std::vector<double>& pressures,
                           const std::vector<std::string>& nodeNames);
    bool initializeSolverSetup(SolverSetup& setup);

    double calculateTotalPressure(double pressure, double temperature, double height) const;
    std::optional<double> calculatePressureDifference(
        const VertexProperties& sourceNode,
        const VertexProperties& targetNode,
        const EdgeProperties& edgeData,
        const PressureMap& pressureMap) const;
    void addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem);
    // 圧力依存有効枝の連結成分ごとに、固定圧境界が無ければ soft anchor を1つ追加
    void addPressureGaugeAnchors(const SolverSetup& setup, ceres::Problem& problem);
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

    // fallback 系は PressureFallbackSolver へ委譲（宣言は互換のため残す）
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
                         const FallbackLogger& fallbackLog);
    SupernodePartition detectSupernodePartition(
        const SimulationConstants& constants,
        const PressureMap& currentPressures);
    StageASolveResult solveStageAReduced(
        const SimulationConstants& constants,
        Graph& g,
        const SupernodePartition& partition,
        const PressureMap& prevPressureMapFB,
        const FallbackLogger& fallbackLog);
    InterfaceFreezeResult freezeInterfaceFlows(
        Graph& g,
        ventilation::EdgeMutationGuard& edgeGuard,
        const SupernodePartition& partition,
        const StageAMapping& stageMapping,
        const PressureMap& pressureMapFB_A,
        const PressureMap& prevPressureMapFB,
        int outer,
        const FallbackLogger& fallbackLog);
    StageBSolveResult solveStageBFull(
        const SimulationConstants& constants,
        Graph& g,
        const SupernodePartition& partition,
        const PressureMap& pressureMapFB_A,
        const FallbackLogger& fallbackLog);
    FallbackOuterAction evaluateFallbackOuter(
        const SimulationConstants& constants,
        Graph& g,
        ventilation::EdgeMutationGuard& edgeGuard,
        const SupernodePartition& partition,
        const StageASolveResult& stageA,
        StageBSolveResult& stageB,
        const InterfaceFreezeResult& freeze,
        int outer,
        int maxOuter,
        int minOuter,
        const std::string& outerTag,
        const ventilation::PressureSolverTolerances& tols,
        FallbackOuterState& state,
        const FallbackLogger& fallbackLog);
    std::optional<SolverResult> runFallbackLoop(
        const SimulationConstants& constants,
        SolverSetup& setup,
        ceres::Solver::Summary& summary);

    std::optional<SolverResult> tryPrimaryWarmStart(
        const SimulationConstants& constants,
        SolverSetup& setup,
        const PressureMap& seedPressures,
        double massBalanceMaxAbs);

    PressureMap extractPressures(const std::vector<double>& pressures,
                                const std::vector<std::string>& nodeNames);

    struct FlowComputationResult {
        FlowRateMap flows;
        bool ok = true;
        std::string detail;
    };

    FlowComputationResult calculateFlowRates(const PressureMap& pressureMap);
    std::optional<std::map<std::string, double>> calculateIndividualFlowRates(const PressureMap& pressureMap);

    ventilation::PressureSolutionEvaluation evaluatePressureSolution(
        const PressureMap& pressureMap,
        double massBalanceMaxAbs);

    ventilation::InterfaceFlowConsistency evaluateInterfaceFlowConsistency(
        const PressureMap& pressureMap,
        const std::vector<std::pair<Edge, double>>& frozenFlows) const;

    FlowBalanceMap verifyBalance(const FlowRateMap& flowRates);

    std::optional<double> calculateFlowForEdge(const PressureMap& pressureMap, Edge edge) const;

    SolverResult solvePressures(const SimulationConstants& constants);
};

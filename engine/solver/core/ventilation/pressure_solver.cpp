#include "core/ventilation/pressure_solver.h"
#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <limits>

double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE / 
           (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

PressureSolver::PressureSolver(VentilationNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

double PressureSolver::calculateTotalPressure(double pressure, double temperature, double height) const {
    double rho = calculateDensity(temperature);
    return pressure - rho * archenv::GRAVITY * height;
}

std::optional<double> PressureSolver::calculatePressureDifference(
    const VertexProperties& sourceNode,
    const VertexProperties& targetNode,
    const EdgeProperties& edgeData,
    const PressureMap& pressureMap) const {
    
    auto sourcePressureIt = pressureMap.find(sourceNode.key);
    auto targetPressureIt = pressureMap.find(targetNode.key);
    
    if (sourcePressureIt == pressureMap.end() || targetPressureIt == pressureMap.end()) {
        return std::nullopt;
    }
    
    double source_total = calculateTotalPressure(
        sourcePressureIt->second, sourceNode.current_t, edgeData.h_from);
    double target_total = calculateTotalPressure(
        targetPressureIt->second, targetNode.current_t, edgeData.h_to);
    
    return source_total - target_total;
}

void PressureSolver::setInitialPressures(std::vector<double>& pressures, 
                                        const std::vector<std::string>& nodeNames) {
    if (pressures.size() != nodeNames.size()) {
        writeLog(logFile_, "--警告: 圧力配列とノード名配列のサイズが一致しません (" + 
                 std::to_string(pressures.size()) + " vs " + std::to_string(nodeNames.size()) + ")");
        return;
    }
    
    const auto& graph = network_.getGraph();
    const auto& keyToVertex = network_.getKeyToVertex();
    for (size_t i = 0; i < pressures.size(); ++i) {
        auto it = keyToVertex.find(nodeNames[i]);
        if (it != keyToVertex.end()) {
            pressures[i] = graph[it->second].current_p;
        } else {
            pressures[i] = 0.0;
        }
    }
}

bool PressureSolver::initializeSolverSetup(SolverSetup& setup) {
    const auto& graph = network_.getGraph();
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    setup.vertexToParameterIndexVec.assign(vCount, -1);

    auto vertex_range = boost::vertices(graph);
    size_t parameterIndex = 0;
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& nodeData = graph[vertex];
        if (nodeData.calc_p) {
            setup.nodeNames.push_back(nodeData.key);
            setup.vertexToParameterIndex[vertex] = parameterIndex++;
            setup.vertexToParameterIndexVec[static_cast<size_t>(vertex)] =
                static_cast<int>(parameterIndex - 1);
        }
    }

    if (setup.nodeNames.empty()) {
        writeLog(logFile_, "--警告: 圧力計算対象のノードがありません");
        return false;
    }

    setup.pressures.resize(setup.nodeNames.size());
    setInitialPressures(setup.pressures, setup.nodeNames);

    setup.incidentEdgesByVertex.assign(vCount, {});
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        Vertex sv = boost::source(e, graph);
        Vertex tv = boost::target(e, graph);
        setup.incidentEdgesByVertex[static_cast<size_t>(sv)].push_back(e);
        setup.incidentEdgesByVertex[static_cast<size_t>(tv)].push_back(e);
    }
    return true;
}

void PressureSolver::addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem) {
    for (const std::string& nodeName : setup.nodeNames) {
        ceres::CostFunction* costFunction = PressureConstraints::createFlowBalanceConstraint(
            nodeName,
            network_.getGraph(),
            network_.getKeyToVertex(),
            setup.vertexToParameterIndexVec,
            setup.incidentEdgesByVertex,
            setup.pressures.size(),
            logFile_);

        problem.AddResidualBlock(costFunction, nullptr, const_cast<double*>(setup.pressures.data()));
    }
}

PressureMap PressureSolver::extractPressures(const std::vector<double>& pressures,
                                            const std::vector<std::string>& nodeNames) {
    PressureMap pressureMap;
    for (size_t i = 0; i < nodeNames.size(); ++i) {
        pressureMap[nodeNames[i]] = pressures[i];
    }
    
    const auto& graph = network_.getGraph();
    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        const auto& properties = graph[vertex];
        if (!properties.calc_p) {
            pressureMap[properties.key] = properties.current_p;
        }
    }
    return pressureMap;
}

FlowBalanceMap PressureSolver::verifyBalance(const FlowRateMap& flowRates) {
    FlowBalanceMap balance;

    const auto& graph = network_.getGraph();

    auto vertex_range = boost::vertices(graph);
    for (auto vertex : boost::make_iterator_range(vertex_range)) {
        balance[graph[vertex].key] = 0.0;
    }

    for (const auto& kv : flowRates) {
        const auto& edgeKey = kv.first;
        const std::string& srcName = edgeKey.first;
        const std::string& dstName = edgeKey.second;
        double q = kv.second;

        if (q == 0.0) continue;

        balance[srcName] -= q;
        balance[dstName] += q;
    }

    return balance;
}

std::optional<double> PressureSolver::calculateFlowForEdge(const PressureMap& pressureMap, Edge edge) const {
    const auto& graph = network_.getGraph();
    auto sv = boost::source(edge, graph);
    auto tv = boost::target(edge, graph);
    const auto& sourceNode = graph[sv];
    const auto& targetNode = graph[tv];
    const auto& edgeData = graph[edge];

    auto dpOpt = calculatePressureDifference(sourceNode, targetNode, edgeData, pressureMap);
    if (!dpOpt) {
        return std::nullopt;
    }
    const double dp = *dpOpt;
    if (!std::isfinite(dp)) {
        return std::nullopt;
    }
    double flow = FlowCalculation::calculateUnifiedFlow(dp, edgeData);
    if (!std::isfinite(flow)) {
        return std::nullopt;
    }
    return flow;
}

PressureSolver::FlowComputationResult PressureSolver::calculateFlowRates(const PressureMap& pressureMap) {
    FlowComputationResult result;
    
    const auto& graph = network_.getGraph();
    auto edge_range = boost::edges(graph);
    
    for (auto edge : boost::make_iterator_range(edge_range)) {
        auto sourceVertex = boost::source(edge, graph);
        auto targetVertex = boost::target(edge, graph);
        
        const auto& sourceNode = graph[sourceVertex];
        const auto& targetNode = graph[targetVertex];
        
        auto flowOpt = calculateFlowForEdge(pressureMap, edge);
        if (!flowOpt) {
            result.ok = false;
            result.detail = "missing_or_nonfinite_flow:" + sourceNode.key + "->" + targetNode.key;
            writeLog(logFile_, "--エラー: 風量評価失敗（圧力欠落または非有限） - " +
                     sourceNode.key + " → " + targetNode.key);
            result.flows.clear();
            return result;
        }
        
        std::pair<std::string, std::string> edgeKey = {sourceNode.key, targetNode.key};
        result.flows[edgeKey] += *flowOpt;
    }
    
    return result;
}

ventilation::PressureSolutionEvaluation PressureSolver::evaluatePressureSolution(
        const PressureMap& pressureMap,
        double massBalanceMaxAbs) {
    ventilation::PressureSolutionEvaluation eval;
    FlowComputationResult flowComp = calculateFlowRates(pressureMap);
    if (!flowComp.ok) {
        eval.flowOk = false;
        eval.accepted = false;
        eval.detail = flowComp.detail;
        return eval;
    }
    eval.flowOk = true;
    eval.flows = std::move(flowComp.flows);
    eval.allNodeBalances = verifyBalance(eval.flows);
    eval.solvedNodeMetrics = ventilation::computePressureUnknownBalanceMetrics(
        eval.allNodeBalances, network_.getGraph());
    eval.accepted = ventilation::acceptMassBalance(eval.solvedNodeMetrics, massBalanceMaxAbs);
    return eval;
}

ventilation::InterfaceFlowConsistency PressureSolver::evaluateInterfaceFlowConsistency(
        const PressureMap& pressureMap,
        const std::vector<std::pair<Edge, double>>& frozenFlows) const {
    ventilation::InterfaceFlowConsistency out;
    out.ok = true;
    out.finite = true;
    for (const auto& item : frozenFlows) {
        const Edge e = item.first;
        const double qFixed = item.second;
        auto qOpt = calculateFlowForEdge(pressureMap, e);
        if (!qOpt || !std::isfinite(*qOpt) || !std::isfinite(qFixed)) {
            out.ok = false;
            out.finite = false;
            return out;
        }
        out.maxAbs = std::max(out.maxAbs, std::abs(*qOpt - qFixed));
        ++out.edgeCount;
    }
    return out;
}

std::optional<PressureSolver::SolverResult> PressureSolver::tryPrimaryWarmStart(
        const SimulationConstants& constants,
        SolverSetup& setup,
        const PressureMap& seedPressures,
        double massBalanceMaxAbs) {
    for (size_t i = 0; i < setup.nodeNames.size(); ++i) {
        auto it = seedPressures.find(setup.nodeNames[i]);
        if (it != seedPressures.end() && std::isfinite(it->second)) {
            setup.pressures[i] = it->second;
        }
    }

    ceres::Problem problem;
    addFlowBalanceConstraints(setup, problem);
    ceres::Solver::Summary summary;
    runPrimarySolvers(constants, problem, summary);

    PressureMap pressureMap = extractPressures(setup.pressures, setup.nodeNames);
    auto eval = evaluatePressureSolution(pressureMap, massBalanceMaxAbs);
    {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "[Fallback] warm-start 評価 | ceres_cost=" << summary.final_cost
            << " | mass_maxAbs=" << (eval.flowOk ? eval.solvedNodeMetrics.maxAbs : -1.0)
            << " | mass_tol=" << massBalanceMaxAbs
            << " | flow_ok=" << (eval.flowOk ? 1 : 0)
            << " | accepted=" << (eval.accepted ? 1 : 0);
        writeLog(logFile_, oss.str());
    }
    if (!eval.flowOk || !eval.accepted) {
        return std::nullopt;
    }
    network_.setLastPressureConverged(true);
    return SolverResult{pressureMap, eval.flows, eval.allNodeBalances};
}

std::optional<std::map<std::string, double>> PressureSolver::calculateIndividualFlowRates(
    const PressureMap& pressureMap) {
    std::map<std::string, double> individualFlowRates;
    
    const auto& graph = network_.getGraph();
    auto edge_range = boost::edges(graph);
    
    for (auto edge : boost::make_iterator_range(edge_range)) {
        const auto& edgeData = graph[edge];
        auto flowOpt = calculateFlowForEdge(pressureMap, edge);
        if (!flowOpt) {
            auto sourceVertex = boost::source(edge, graph);
            auto targetVertex = boost::target(edge, graph);
            writeLog(logFile_, "--エラー: 個別風量評価失敗 - " + edgeData.unique_id +
                     " (" + graph[sourceVertex].key + " → " + graph[targetVertex].key + ")");
            return std::nullopt;
        }
        individualFlowRates[edgeData.unique_id] = *flowOpt;
    }
    
    return individualFlowRates;
}

PressureSolver::SolverResult PressureSolver::solvePressures(
    const SimulationConstants& constants) {
    const auto tols = ventilation::makePressureSolverTolerances(constants);

    SolverSetup setup;
    if (!initializeSolverSetup(setup)) {
        network_.setLastPressureConverged(false);
        return SolverResult{PressureMap{}, FlowRateMap{}, FlowBalanceMap{}};
    }
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

    ceres::Problem problem;
    addFlowBalanceConstraints(setup, problem);

    ceres::Solver::Summary summary;
    runPrimarySolvers(constants, problem, summary);

    PressureMap pressureMap = extractPressures(pressures, nodeNames);
    auto eval = evaluatePressureSolution(pressureMap, tols.massBalanceMaxAbs);
    if (!eval.flowOk) {
        writeLog(logFile_, "--警告: primary 解の風量評価に失敗。fallback を試行します。");
        network_.setLastPressureConverged(false);
        auto fallbackResult = runFallbackLoop(constants, setup, summary);
        if (fallbackResult) {
            return *fallbackResult;
        }
        return SolverResult{pressureMap, FlowRateMap{}, FlowBalanceMap{}};
    }

    const auto& metrics = eval.solvedNodeMetrics;
    const bool massAccepted = eval.accepted;
    const bool ceresSaidConverged = (summary.termination_type == ceres::CONVERGENCE);

    {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6);
        oss << "---圧力計算評価 | ceres_term="
            << (ceresSaidConverged ? "CONVERGENCE" : "OTHER")
            << " | ceres_cost=" << summary.final_cost
            << " | mass_maxAbs=" << metrics.maxAbs
            << " | mass_tol=" << tols.massBalanceMaxAbs
            << " | solved_nodes=" << metrics.nodeCount
            << " | iter=" << summary.iterations.size();
        writeLog(logFile_, oss.str());
    }

    if (massAccepted) {
        network_.setLastPressureConverged(true);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << metrics.maxAbs;
        writeLog(logFile_,
                 "---圧力計算収束 | mass_maxAbs=" + oss.str() +
                     " | tol=" + std::to_string(tols.massBalanceMaxAbs));
        return SolverResult{pressureMap, eval.flows, eval.allNodeBalances};
    }

    auto fallbackResult = runFallbackLoop(constants, setup, summary);
    if (fallbackResult) {
        return *fallbackResult;
    }

    network_.setLastPressureConverged(false);
    return SolverResult{pressureMap, eval.flows, eval.allNodeBalances};
}

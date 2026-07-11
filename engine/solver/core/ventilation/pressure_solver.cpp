#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/flow_calculation.h"
#include "core/ventilation/pressure_balance.h"
#include "core/ventilation/pressure_constraints.h"
#include "core/ventilation/pressure_solver_internal.h"
#include "network/ventilation_network.h"
#include "utils/utils.h"
#include "../archenv/include/archenv.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <limits>

double calculateDensity(double temperature) {
    return archenv::STANDARD_ATMOSPHERIC_PRESSURE / 
           (archenv::GAS_CONSTANT_DRY_AIR * (temperature + 273.15));
}

PressureSolver::Impl::Impl(VentilationNetwork& network, std::ostream& logFile)
    : network_(network), logFile_(logFile) {}

PressureSolver::PressureSolver(VentilationNetwork& network, std::ostream& logFile)
    : impl_(std::make_unique<Impl>(network, logFile)) {}

PressureSolver::~PressureSolver() = default;

PressureSolver::PressureSolver(PressureSolver&&) noexcept = default;
PressureSolver& PressureSolver::operator=(PressureSolver&&) noexcept = default;

PressureSolveResult PressureSolver::solveDetailed(const SimulationConstants& constants) {
    return impl_->solvePressures(constants);
}

PressureSolver::SolverResult PressureSolver::solvePressures(const SimulationConstants& constants) {
    return solveDetailed(constants).asTuple();
}

double PressureSolver::Impl::calculateTotalPressure(double pressure, double temperature, double height) const {
    double rho = calculateDensity(temperature);
    return pressure - rho * archenv::GRAVITY * height;
}

std::optional<double> PressureSolver::Impl::calculatePressureDifference(
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

void PressureSolver::Impl::setInitialPressures(std::vector<double>& pressures, 
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

bool PressureSolver::Impl::initializeSolverSetup(SolverSetup& setup) {
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

void PressureSolver::Impl::addFlowBalanceConstraints(const SolverSetup& setup, ceres::Problem& problem) {
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
    addPressureGaugeAnchors(setup, problem);
}

void PressureSolver::Impl::addPressureGaugeAnchors(const SolverSetup& setup, ceres::Problem& problem) {
    addPressureGaugeAnchors(network_.getGraph(),
                            setup.vertexToParameterIndexVec,
                            setup.pressures,
                            &setup.incidentEdgesByVertex,
                            problem,
                            /*anchorWeight=*/1.0);
}

void PressureSolver::Impl::addPressureGaugeAnchors(
        const Graph& graph,
        const std::vector<int>& vertexToParameterIndexVec,
        const std::vector<double>& pressures,
        const std::vector<std::vector<Edge>>* incidentEdgesByVertex,
        ceres::Problem& problem,
        double anchorWeight) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    if (vCount == 0 || pressures.empty()) {
        return;
    }

    auto isPressureCouplingEdge = [](const EdgeProperties& ep) {
        if (!ep.current_enabled || FlowCalculation::isFixedVolumeFlowEdge(ep)) {
            return false;
        }
        return ep.type == "simple_opening" || ep.type == "gap" ||
               ep.type == "fan" || ep.type == "pressure_loss";
    };

    std::vector<int> parent(static_cast<int>(vCount));
    for (int i = 0; i < static_cast<int>(vCount); ++i) {
        parent[static_cast<size_t>(i)] = i;
    }
    std::function<int(int)> find = [&](int x) {
        if (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = find(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[static_cast<size_t>(b)] = a;
        }
    };

    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        if (!isPressureCouplingEdge(graph[e])) {
            continue;
        }
        unite(static_cast<int>(boost::source(e, graph)), static_cast<int>(boost::target(e, graph)));
    }

    struct CompInfo {
        bool hasFixedPressure = false;
        bool hasUnknown = false;
        int anchorParam = -1;
        double anchorTarget = 0.0;
        std::string anchorKey;
        bool hasPressureCoupling = false;
        double fixedFlowBalanceAbsMax = 0.0;
    };
    std::unordered_map<int, CompInfo> comps;

    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        if (!isPressureCouplingEdge(graph[e])) {
            continue;
        }
        comps[find(static_cast<int>(boost::source(e, graph)))].hasPressureCoupling = true;
    }

    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const int root = find(static_cast<int>(v));
        auto& info = comps[root];
        const auto& node = graph[v];
        if (!node.calc_p) {
            info.hasFixedPressure = true;
            continue;
        }
        const int param = (static_cast<size_t>(v) < vertexToParameterIndexVec.size())
                              ? vertexToParameterIndexVec[static_cast<size_t>(v)]
                              : -1;
        if (param < 0) {
            continue;
        }
        info.hasUnknown = true;
        if (info.anchorParam < 0) {
            info.anchorParam = param;
            info.anchorTarget = (static_cast<size_t>(param) < pressures.size() &&
                                 std::isfinite(pressures[static_cast<size_t>(param)]))
                                    ? pressures[static_cast<size_t>(param)]
                                    : (std::isfinite(node.current_p) ? node.current_p : 0.0);
            info.anchorKey = node.key;
        }
    }

    if (incidentEdgesByVertex != nullptr) {
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            const auto& node = graph[v];
            if (!node.calc_p) {
                continue;
            }
            const int root = find(static_cast<int>(v));
            auto it = comps.find(root);
            if (it == comps.end() || it->second.hasPressureCoupling) {
                continue;
            }
            double bal = 0.0;
            if (static_cast<size_t>(v) < incidentEdgesByVertex->size()) {
                for (Edge e : (*incidentEdgesByVertex)[static_cast<size_t>(v)]) {
                    const auto& ep = graph[e];
                    if (!ep.current_enabled || !FlowCalculation::isFixedVolumeFlowEdge(ep)) {
                        continue;
                    }
                    const Vertex sv = boost::source(e, graph);
                    const double q = ep.current_vol;
                    bal += (sv == v) ? -q : q;
                }
            }
            it->second.fixedFlowBalanceAbsMax =
                std::max(it->second.fixedFlowBalanceAbsMax, std::abs(bal));
        }
    }

    int anchorsAdded = 0;
    for (const auto& kv : comps) {
        const CompInfo& info = kv.second;
        if (!info.hasUnknown) {
            continue;
        }
        if (!info.hasPressureCoupling) {
            if (info.fixedFlowBalanceAbsMax > 1e-12) {
                writeLog(logFile_,
                         "--警告: 固定流量のみの成分で体積流量収支が不整合 | maxAbs=" +
                             std::to_string(info.fixedFlowBalanceAbsMax));
            }
            continue;
        }
        if (info.hasFixedPressure || info.anchorParam < 0) {
            continue;
        }
        problem.AddResidualBlock(
            PressureConstraints::createSoftAnchorConstraint(
                static_cast<size_t>(info.anchorParam),
                info.anchorTarget,
                anchorWeight,
                pressures.size()),
            nullptr,
            const_cast<double*>(pressures.data()));
        ++anchorsAdded;
        writeLog(logFile_,
                 "--圧力ゲージ固定: 成分に固定圧境界がないため soft anchor を追加 | node=" +
                     info.anchorKey + " | target_p=" + std::to_string(info.anchorTarget));
    }
    if (anchorsAdded > 0) {
        writeLog(logFile_,
                 "--圧力ゲージ固定: soft anchor 数=" + std::to_string(anchorsAdded));
    }
}

PressureMap PressureSolver::Impl::extractPressures(const std::vector<double>& pressures,
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

FlowBalanceMap PressureSolver::Impl::verifyBalance(const FlowRateMap& flowRates) {
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

std::optional<double> PressureSolver::Impl::calculateFlowForEdge(const PressureMap& pressureMap, Edge edge) const {
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

PressureSolver::Impl::FlowComputationResult PressureSolver::Impl::calculateFlowRates(const PressureMap& pressureMap) {
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

ventilation::PressureSolutionEvaluation PressureSolver::Impl::evaluatePressureSolution(
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

ventilation::InterfaceFlowConsistency PressureSolver::Impl::evaluateInterfaceFlowConsistency(
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

std::optional<PressureSolver::Impl::SolverResult> PressureSolver::Impl::tryPrimaryWarmStart(
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
    bool physicalAccepted = false;
    runPrimarySolvers(constants, problem, summary, setup, massBalanceMaxAbs, physicalAccepted);

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
    return makePressureSolveResult(
        pressureMap, eval.flows, eval.allNodeBalances,
        /*accepted=*/true, eval.solvedNodeMetrics, "fallback_warmstart");
}

std::optional<std::map<std::string, double>> PressureSolver::Impl::calculateIndividualFlowRates(
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

PressureSolver::Impl::SolverResult PressureSolver::Impl::solvePressures(
    const SimulationConstants& constants) {
    const auto tols = ventilation::makePressureSolverTolerances(constants);

    SolverSetup setup;
    if (!initializeSolverSetup(setup)) {
        // calc_p ノードなし = 全圧既知。流量のみ評価して自明解として受理する。
        writeLog(logFile_, "--情報: 圧力未知ノードがないため固定圧ネットワークとして流量評価します");
        PressureMap pressureMap;
        const auto& graph = network_.getGraph();
        for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
            pressureMap[graph[v].key] = graph[v].current_p;
        }
        auto eval = evaluatePressureSolution(pressureMap, tols.massBalanceMaxAbs);
        if (!eval.flowOk) {
            network_.setLastPressureConverged(false);
            writeLog(logFile_, "--警告: 固定圧ネットワークの風量評価に失敗: " + eval.detail);
            return makePressureSolveResult(pressureMap, {}, {}, /*accepted=*/false, {}, "fixed_pressure");
        }
        // 未知圧が無い場合、質量収支の「計算ノード」は空 → acceptMassBalance は complete=false になり得る。
        // 流量が有限に評価できれば自明解として受理する。
        network_.setLastPressureConverged(true);
        writeLog(logFile_, "---固定圧ネットワーク: 流量評価完了（圧力求解スキップ）");
        return makePressureSolveResult(
            pressureMap, eval.flows, eval.allNodeBalances,
            /*accepted=*/true, eval.solvedNodeMetrics, "fixed_pressure");
    }
    auto& nodeNames = setup.nodeNames;
    auto& pressures = setup.pressures;

    ceres::Problem problem;
    addFlowBalanceConstraints(setup, problem);

    ceres::Solver::Summary summary;
    bool physicalAccepted = false;
    runPrimarySolvers(constants, problem, summary, setup, tols.massBalanceMaxAbs, physicalAccepted);

    PressureMap pressureMap = extractPressures(pressures, nodeNames);
    auto eval = evaluatePressureSolution(pressureMap, tols.massBalanceMaxAbs);
    if (!eval.flowOk) {
        writeLog(logFile_, "--警告: primary 解の風量評価に失敗。fallback を試行します。");
        network_.setLastPressureConverged(false);
        auto fallbackResult = runFallbackLoop(constants, setup, summary);
        if (fallbackResult) {
            return *fallbackResult;
        }
        return makePressureSolveResult(pressureMap, {}, {}, /*accepted=*/false, {}, "");
    }

    const auto& metrics = eval.solvedNodeMetrics;
    const bool massAccepted = physicalAccepted || eval.accepted;
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
            << " | physical_accepted=" << (massAccepted ? 1 : 0)
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
        return makePressureSolveResult(
            pressureMap, eval.flows, eval.allNodeBalances,
            /*accepted=*/true, metrics, "primary");
    }

    auto fallbackResult = runFallbackLoop(constants, setup, summary);
    if (fallbackResult) {
        return *fallbackResult;
    }

    network_.setLastPressureConverged(false);
    return makePressureSolveResult(
        pressureMap, eval.flows, eval.allNodeBalances,
        /*accepted=*/false, metrics, "");
}

#include "core/ventilation/pressure_solver_impl.h"
#include "core/ventilation/edge_mutation_guard.h"
#include "core/ventilation/pressure_solver_internal.h"

#include <map>

PressureSolver::Impl::InterfaceFreezeResult PressureSolver::Impl::freezeInterfaceFlows(
        Graph& g,
        ventilation::EdgeMutationGuard& edgeGuard,
        const SupernodePartition& partition,
        const StageAMapping& stageMapping,
        const PressureMap& pressureMapFB_A,
        const PressureMap& prevPressureMapFB,
        int outer,
        const FallbackLogger& fallbackLog) {
    InterfaceFreezeResult result;
    const auto& vToParamIdx = stageMapping.vertexToParamIndex;
    const auto& v2i = partition.v2i;
    const auto& groupOfVertex = partition.groupOfVertex;

    PressureMap pressureMapForFixed = (outer >= 2) ? prevPressureMapFB : pressureMapFB_A;
    auto indivOpt = calculateIndividualFlowRates(pressureMapForFixed);
    if (!indivOpt) {
        fallbackLog(2, "[A] 固定流量化スキップ: 風量評価失敗");
        result.skipped = true;
        return result;
    }
    std::map<std::string, double> indivA = std::move(*indivOpt);

    auto erInt = boost::edges(g);
    for (auto e : boost::make_iterator_range(erInt)) {
        auto sv = boost::source(e, g);
        auto tv = boost::target(e, g);
        auto itS = vToParamIdx.find(sv);
        auto itT = vToParamIdx.find(tv);
        if (itS != vToParamIdx.end() && itT != vToParamIdx.end() && itS->second == itT->second) {
            const auto& epz = g[e];
            auto itId = indivA.find(epz.unique_id);
            if (itId != indivA.end()) itId->second = 0.0;
        }
    }

    auto er2 = boost::edges(g);
    for (auto e : boost::make_iterator_range(er2)) {
        auto sv = boost::source(e, g);
        auto tv = boost::target(e, g);
        auto& ep = g[e];

        int si = v2i.at(sv);
        int ti = v2i.at(tv);
        int gidS = groupOfVertex[si];
        int gidT = groupOfVertex[ti];
        bool crossCluster = (gidS != gidT) && (gidS >= 0 || gidT >= 0);
        if (!crossCluster) continue;
        if (ep.type == "fixed_flow") {
            result.alreadyFixed++;
            continue;
        }

        auto itf = indivA.find(ep.unique_id);
        if (itf != indivA.end()) {
            edgeGuard.convertToFixedFlow(e, itf->second);
            result.frozenFlows.emplace_back(e, itf->second);
            result.fixedFlowCount++;
        }
    }
    const std::string flowSource = (outer >= 2) ? "source=B(prev)" : "source=A(current)";
    fallbackLog(2, "[A] 固定流量化ブランチ=" + std::to_string(result.fixedFlowCount) +
                      " | already_fixed=" + std::to_string(result.alreadyFixed) +
                      " | " + flowSource);
    return result;
}

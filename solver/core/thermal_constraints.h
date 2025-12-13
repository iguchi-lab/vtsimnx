#pragma once

#include "../vtsim_solver.h"
#include "core/heat_calculation.h"
#include <ceres/ceres.h>
#include <map>
#include <ostream>
#include <unordered_map>
#include <vector>

// =============================================================================
// HeatBalanceConstraint - 熱バランス制約（解析ヤコビアン版）
// 方式は換気（圧力）の制約と同様に、inflow/outflow を用いた有向和で統一
// =============================================================================
class HeatBalanceConstraint : public ceres::CostFunction {
public:
	HeatBalanceConstraint(
	    const std::string& nodeName,
	    const Graph& graph,
	    const std::unordered_map<std::string, Vertex>& nodeKeyToVertex,
	    const std::map<Vertex, size_t>& vertexToParameterIndex,
	    const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges,
	    const std::unordered_map<std::string, std::vector<Vertex>>& airconBySetNode,
	    size_t numParameters,
	    std::ostream& logFile)
	    : nodeName_(nodeName),
	      graph_(graph),
	      nodeKeyToVertex_(nodeKeyToVertex),
	      vertexToParameterIndex_(vertexToParameterIndex),
	      incidentEdges_(incidentEdges),
	      airconBySetNode_(airconBySetNode),
	      numParameters_(numParameters),
	      logFile_(logFile)
	{
		set_num_residuals(1);
		mutable_parameter_block_sizes()->push_back(static_cast<int>(numParameters_));
	}

	bool Evaluate(double const* const* parameters,
	              double* residuals,
	              double** jacobians) const override
	{
		auto nodeIt = nodeKeyToVertex_.find(nodeName_);
		if (nodeIt == nodeKeyToVertex_.end()) {
			residuals[0] = 0.0;
			if (jacobians && jacobians[0]) {
				std::fill(jacobians[0], jacobians[0] + static_cast<int>(numParameters_), 0.0);
			}
			return true;
		}
		const double* temps = parameters[0];
		Vertex nodeVertex = nodeIt->second;

		auto zeroJac = [&]() {
			if (jacobians && jacobians[0]) {
				std::fill(jacobians[0], jacobians[0] + static_cast<int>(numParameters_), 0.0);
			}
		};

		// set_node がエアコンONなら常に残差0（ヤコビアンも0）
		auto itSetAc = airconBySetNode_.find(nodeName_);
		if (itSetAc != airconBySetNode_.end()) {
			for (auto v_ac : itSetAc->second) {
				const auto& nd = graph_[v_ac];
				if (nd.type == "aircon" && nd.on && nd.set_node == nodeName_) {
					residuals[0] = 0.0;
					zeroJac();
					return true;
				}
			}
		}

		double inflow = 0.0;
		double outflow = 0.0;
		std::vector<double> grad(numParameters_, 0.0);

		const auto& nodeEdges = getIncidentEdges(nodeVertex);
		for (auto edge : nodeEdges) {
			Vertex sv = boost::source(edge, graph_);
			Vertex tv = boost::target(edge, graph_);
			const auto& eprop = graph_[edge];

			// 温度取得
			auto itS = vertexToParameterIndex_.find(sv);
			auto itT = vertexToParameterIndex_.find(tv);
			double Ts = (itS != vertexToParameterIndex_.end()) ? temps[itS->second] : graph_[sv].current_t;
			double Tt = (itT != vertexToParameterIndex_.end()) ? temps[itT->second] : graph_[tv].current_t;

			// Q と dQ/dTs, dQ/dTt
			double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, eprop);
			double dQdTs = 0.0;
			double dQdTt = 0.0;
			if (eprop.type == "conductance") {
				dQdTs = eprop.conductance;
				dQdTt = -eprop.conductance;
			} else if (eprop.type == "advection") {
				double mDotCpAbs = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(eprop.flow_rate);
				dQdTs = mDotCpAbs;
				dQdTt = -mDotCpAbs;
			} else if (eprop.type == "heat_generation") {
				dQdTs = 0.0;
				dQdTt = 0.0;
			}

			if (sv == nodeVertex) {
				outflow += Q;
				if (itS != vertexToParameterIndex_.end()) grad[itS->second] -= dQdTs;
				if (itT != vertexToParameterIndex_.end()) grad[itT->second] -= dQdTt;
			} else if (tv == nodeVertex) {
				inflow += Q;
				if (itS != vertexToParameterIndex_.end()) grad[itS->second] += dQdTs;
				if (itT != vertexToParameterIndex_.end()) grad[itT->second] += dQdTt;
			}
		}

		// エアコンノードなら set_node の net を肩代わり（r += -net_set）
		const auto& nodeData = graph_[nodeVertex];
		if (nodeData.type == "aircon" && nodeData.on && !nodeData.set_node.empty()) {
			auto itSet = nodeKeyToVertex_.find(nodeData.set_node);
			if (itSet != nodeKeyToVertex_.end()) {
				Vertex setV = itSet->second;
				double setIn = 0.0;
				double setOut = 0.0;

				const auto& setEdges = getIncidentEdges(setV);
				for (auto e2 : setEdges) {
					Vertex sv = boost::source(e2, graph_);
					Vertex tv = boost::target(e2, graph_);
					const auto& ep = graph_[e2];
					auto is = vertexToParameterIndex_.find(sv);
					auto it = vertexToParameterIndex_.find(tv);
					double Ts = (is != vertexToParameterIndex_.end()) ? temps[is->second] : graph_[sv].current_t;
					double Tt = (it != vertexToParameterIndex_.end()) ? temps[it->second] : graph_[tv].current_t;

					double Q = HeatCalculation::calculateUnifiedHeat(Ts, Tt, ep);
					double dQdTs = 0.0, dQdTt = 0.0;
					if (ep.type == "conductance") {
						dQdTs = ep.conductance;
						dQdTt = -ep.conductance;
					} else if (ep.type == "advection") {
						double mDotCpAbs = archenv::DENSITY_DRY_AIR * archenv::SPECIFIC_HEAT_AIR * std::abs(ep.flow_rate);
						dQdTs = mDotCpAbs;
						dQdTt = -mDotCpAbs;
					}
					if (sv == setV) {
						setOut += Q;
						if (is != vertexToParameterIndex_.end()) grad[is->second] += dQdTs;   // −(−dQ) = +dQ
						if (it != vertexToParameterIndex_.end()) grad[it->second] += dQdTt;   // 同上
					} else if (tv == setV) {
						setIn += Q;
						if (is != vertexToParameterIndex_.end()) grad[is->second] -= dQdTs;   // −(+dQ) = −dQ
						if (it != vertexToParameterIndex_.end()) grad[it->second] -= dQdTt;
					}
				}
				// inflow - outflow に対して −(setIn - setOut) を加える → 勾配は上で反映済み
				inflow += -(setIn - setOut);
			}
		}

		residuals[0] = inflow - outflow;

		if (jacobians && jacobians[0]) {
			std::copy(grad.begin(), grad.end(), jacobians[0]);
		}
		return true;
	}

private:
	std::string nodeName_;
	const Graph& graph_;
	const std::unordered_map<std::string, Vertex>& nodeKeyToVertex_;
	const std::map<Vertex, size_t>& vertexToParameterIndex_;
	const std::unordered_map<Vertex, std::vector<Edge>>& incidentEdges_;
	const std::unordered_map<std::string, std::vector<Vertex>>& airconBySetNode_;
	size_t numParameters_;
	std::ostream& logFile_;

	const std::vector<Edge>& getIncidentEdges(Vertex v) const {
		static const std::vector<Edge> kEmpty;
		auto it = incidentEdges_.find(v);
		if (it != incidentEdges_.end()) return it->second;
		return kEmpty;
	}
};



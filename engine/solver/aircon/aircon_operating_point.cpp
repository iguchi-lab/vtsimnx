#include "aircon/aircon_controller.h"

#include "aircon/aircon_airflow.h"
#include "aircon/aircon_humidity_control.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_network_utils.h"
#include "network/thermal_network.h"
#include "utils/utils.h"

namespace {
void copyRuntime(const VertexProperties& a, VertexProperties& b) {
    b.current_p=a.current_p; b.current_t=a.current_t; b.current_x=a.current_x;
    b.current_w=a.current_w; b.heat_source=a.heat_source;
    b.current_pre_temp=a.current_pre_temp; b.required_heat_w=a.required_heat_w;
    b.on=a.on; b.aircon_control_state=a.aircon_control_state;
    b.aircon_flow_unmet_boost=a.aircon_flow_unmet_boost;
    b.aircon_min_process_hold=a.aircon_min_process_hold;
    b.aircon_moisture_removal_kg_s=a.aircon_moisture_removal_kg_s;
    b.humidity_supply_request=a.humidity_supply_request;
    b.humidity_demand=a.humidity_demand; b.humidity_overcool_hold=a.humidity_overcool_hold;
    b.humidity_capacity_limited=a.humidity_capacity_limited;
    b.humidity_search_low=a.humidity_search_low; b.humidity_search_high=a.humidity_search_high;
    b.humidity_search_error_low=a.humidity_search_error_low;
    b.humidity_search_error_high=a.humidity_search_error_high;
}
void copyRuntime(const EdgeProperties& a, EdgeProperties& b) {
    b.flow_rate=a.flow_rate; b.current_vol=a.current_vol;
    b.p_max=a.p_max; b.p1=a.p1; b.q_max=a.q_max; b.q1=a.q1;
    b.fan_curve_rated=a.fan_curve_rated;
    b.p_max_rated=a.p_max_rated; b.p1_rated=a.p1_rated;
    b.q_max_rated=a.q_max_rated; b.q1_rated=a.q1_rated;
}
double designFlow(const VertexProperties& node) {
    const auto& spec = node.ac_spec;
    if (!spec.contains("V_inner") || !spec["V_inner"].contains("cooling")) return 0.0;
    return spec["V_inner"]["cooling"].value("dsgn", 0.0);
}

double currentRatio(const VentilationNetwork& network, const VertexProperties& node,
                    double actualFlow, double design) {
    const auto& graph = network.getGraph();
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        const auto& edge = graph[e];
        if (edge.type == "fan" && edge.subtype == "aircon" &&
            (graph[boost::source(e, graph)].key == node.key ||
             graph[boost::target(e, graph)].key == node.key)) {
            const double rated = edge.fan_curve_rated ? edge.q_max_rated : edge.q_max;
            return rated > 0.0 ? std::clamp(edge.q_max/rated, 0.0, 1.0) : 0.0;
        }
    }
    return std::clamp(std::abs(actualFlow)/design, 0.0, 1.0);
}
} // namespace

void AirconController::captureAirflowCandidate(ThermalNetwork& thermal,
        VentilationNetwork& ventilation, const std::string& key, double ratio) const {
    airflowCandidateSnapshot_.emplace();
    auto& snap=*airflowCandidateSnapshot_;
    snap.key=key; snap.ratio=ratio;
    auto capture=[](const Graph& graph, std::vector<VertexProperties>& nodes) {
        nodes.resize(boost::num_vertices(graph));
        for (auto v:boost::make_iterator_range(boost::vertices(graph))) copyRuntime(graph[v],nodes[v]);
    };
    capture(thermal.getGraph(),snap.thermalNodes);
    capture(ventilation.getGraph(),snap.ventilationNodes);
    for (auto e:boost::make_iterator_range(boost::edges(ventilation.getGraph()))) {
        snap.ventilationEdges.emplace_back();
        copyRuntime(ventilation.getGraph()[e],snap.ventilationEdges.back());
    }
    snap.capacity=capacityLimitBracket_; snap.flowBoost=capacityFlowBoost_;
    snap.recovery=capacityRecoveryAttempted_; snap.floorBlocked=humidityFloorBlocked_;
    snap.fanTrial=humidityFanTrialDone_; snap.rebase=humiditySearchRebase_;
    snap.boundary=boundaryIterations_; snap.freeTemperature=lastFreeSetTempC_;
}

bool AirconController::rejectHumidityAirflowCandidate(ThermalNetwork& thermal,
        VentilationNetwork& ventilation, std::ostream& logs, const std::string& detail) const {
    if (!airflowCandidateSnapshot_) return false;
    for (auto v:boost::make_iterator_range(boost::vertices(thermal.getGraph())))
        if (!std::isfinite(thermal.getGraph()[v].current_t)) return false;
    auto& snap=*airflowCandidateSnapshot_;
    if (!snap.initialWarmStart) {
        auto& search=humidityAirflowSearch_.at(snap.key);
        if (search.complete || search.restoring) return false;
        writeDomainLog(logs,"空調運転点",snap.key+" rejected speculative airflow ratio="+
            std::to_string(search.ratio)+": "+detail);
        search.tried.push_back(search.ratio);
    } else {
        writeDomainLog(logs,"空調運転点","rejected initial humidity warm start: "+detail);
    }
    auto restore=[](Graph& graph,const std::vector<VertexProperties>& nodes) {
        for (auto v:boost::make_iterator_range(boost::vertices(graph))) copyRuntime(nodes[v],graph[v]);
    };
    restore(thermal.getGraph(),snap.thermalNodes);
    restore(ventilation.getGraph(),snap.ventilationNodes);
    size_t index=0;
    for (auto e:boost::make_iterator_range(boost::edges(ventilation.getGraph())))
        copyRuntime(snap.ventilationEdges[index++],ventilation.getGraph()[e]);
    capacityLimitBracket_=snap.capacity; capacityFlowBoost_=snap.flowBoost;
    capacityRecoveryAttempted_=snap.recovery; humidityFloorBlocked_=snap.floorBlocked;
    humidityFanTrialDone_=snap.fanTrial; humiditySearchRebase_=snap.rebase;
    boundaryIterations_=snap.boundary; lastFreeSetTempC_=snap.freeTemperature;
    if (snap.initialWarmStart) {
        // Previous airflow is only an initial guess. Retry the canonical
        // full-flow initial point once, from the unchanged timestep inventory.
        disableHumidityWarmStart_=true;
        humidityAirflowSearch_.clear();
        airflowCandidateSnapshot_.reset();
        thermal.syncFlowRatesFromVentilationNetwork(ventilation);
        return true;
    }
    humidityAirflowSearch_.at(snap.key).ratio=snap.ratio;
    thermal.syncFlowRatesFromVentilationNetwork(ventilation);
    advanceHumidityAirflowSearch(thermal,ventilation,ventilation.collectFlowRateMap(),logs);
    return true; // Always solve/validate again before exporting any restored state.
}

aircon::operating_point::CoolingRequest AirconController::coolingRequest(
        ThermalNetwork& thermal, const VertexProperties& node,
        bool temperatureDemand, bool numericalTrial, bool mayCool, bool blocked,
        const FlowRateMap* flows) const {
    aircon::operating_point::CoolingRequest request;
    request.temperature = temperatureDemand;
    request.humidity = node.humidity_demand;
    request.numericalTrial = numericalTrial;
    request.mayCool = mayCool;
    request.blocked = blocked;
    const double roomT = aircon::network_utils::getControlledRoomTemperature(thermal, node);
    request.reheat = aircon::humidity_control::simultaneous(node) &&
        (roomT < node.current_requested_pre_temp - aircon::onoff::kRestartBandK ||
         (node.on && std::isfinite(node.required_heat_w) && node.required_heat_w > 1.0));
    if (flows && node.on) {
        const auto context = prepareRuntimeContext(node.key, thermal, node, *flows);
        const auto loads = aircon::latent::estimateLatentProcess(context.validData,
            OperationMode::Cooling, context.heatCapacity, context.airFlowRate, node,
            moistEnthalpyEnabled_);
        request.sensibleW = loads.sensibleHeatCapacity;
        request.latentW = loads.latentHeatCapacity;
        if (request.humidity && std::isfinite(node.humidity_supply_request)) {
            const double water = archenv::DENSITY_DRY_AIR*std::abs(context.airFlowRate)*std::max(0.0,
                context.validData.indoorX-node.humidity_supply_request);
            request.latentW = std::max(request.latentW,
                water*archenv::vapor_latent_heat(context.validData.airconTemp));
        }
    }
    return request;
}

std::string AirconController::invalidMixedSupplyOutlet(ThermalNetwork& thermal,
                                                       const FlowRateMap& flows) const {
    for (const auto& key : getAirconKeys()) {
        const auto& node = thermal.getNode(key);
        if (!node.on) continue;
        const auto context = prepareRuntimeContext(key, thermal, node, flows);
        if (context.operationMode != OperationMode::Cooling) continue;
        const auto pure = aircon::network_utils::passiveSupplyOutletConstraints(thermal, node);
        for (const auto& outletKey : aircon::network_utils::supplyOutletKeys(thermal, node)) {
            if (std::any_of(pure.begin(), pure.end(), [&](const auto& c) { return c.outletKey == outletKey; }))
                continue; // Pure supply already has a conservative upstream boundary.
            const auto& outlet = thermal.getNode(outletKey);
            if (!aircon::latent::unsaturatedBeforeAndAfterSerialization(outlet.current_t, outlet.current_x))
                return "Supersaturated mixed supply outlet: " + outletKey;
        }
    }
    return {};
}

bool AirconController::advanceHumidityAirflowSearch(ThermalNetwork& thermal,
        VentilationNetwork& ventilation, const FlowRateMap& flows, std::ostream& logs) const {
    // Called only after the complete coupled candidate passed physical checks.
    // Coordinate one equipment search at a time; other units retain their
    // airflow while their temperature/humidity/capacity controls are re-solved.
    for (const auto& key : getAirconKeys()) {
        auto& node = thermal.getNode(key);
        if (!aircon::airflow::isDuctCentralModel(node) ||
            !aircon::humidity_control::enabled(node)) continue;
        const double design = designFlow(node);
        if (!(design > 0.0) || !std::isfinite(design)) continue;
        auto& search = humidityAirflowSearch_[key];
        if (search.complete) continue;
        const auto& room = thermal.getNode(node.set_node);
        const double vapor = archenv::STANDARD_ATMOSPHERIC_PRESSURE * room.current_x /
            (0.622 + room.current_x);
        const double rh = 100.0*vapor/archenv::saturation_vapor_pressure(room.current_t);
        const aircon::operating_point::Score score{
            std::max(0.0, room.current_t-node.current_requested_pre_temp),
            std::max(0.0, rh-node.current_pre_rh)};
        const double actualFlow = aircon::network_utils::getAirconProcessFlowRate(flows, node.in_node, key);
        auto logSelection = [&]() {
            if (node.on) humidityAirflowWarmRatio_[key] = search.ratio;
            else humidityAirflowWarmRatio_.erase(key);
            const auto power = estimatePowerAndCOPForAircon(key, thermal, node, flows, logs, false).first;
            std::ostringstream message;
            message << key << " selected airflow ratio=" << search.ratio
                << ", actual_flow_m3s=" << actualFlow << ", on=" << node.on
                << ", room_T_C=" << room.current_t << ", room_RH=" << rh
                << ", humidity_unmet_pp=" << score.humidityExcess
                << ", power_W=" << power << ", requirements_met=" << score.feasible();
            writeDomainLog(logs, "空調運転点", message.str());
        };
        if (search.restoring) {
            search.complete = true;
            logSelection();
            continue;
        }
        const bool solvedAlternative = search.initialized;
        if (!search.initialized) {
            if (!node.on && !node.humidity_demand) { search.complete = true; continue; }
            if (!node.on && !aircon::humidity_control::simultaneous(node) &&
                room.current_t < node.current_requested_pre_temp-node.max_overcooling-1e-4) {
                // An already cold, stopped room cannot start cooling to reach
                // its lower bound. Do not manufacture heating for a fan trial.
                search.complete = true;
                continue;
            }
            search.initialized = true;
            search.ratio = currentRatio(ventilation, node, actualFlow, design);
            const auto* spec = node.getAirconSpec();
            const double minimumW = spec ? 1000.0*spec->getMinProcess("cooling").value_or(0.0) : 0.0;
            const auto minimumFlow = aircon::airflow::computeTargetFlowFromProcessedHeat(
                node, OperationMode::Cooling, std::max(1.0, minimumW), 0.0, minimumW);
            if (!minimumFlow) { search.complete = true; continue; }
            // Same declared control floor as temperature-priority operation;
            // this is not a newly invented hardware minimum airflow.
            search.minimumRatio = std::clamp(*minimumFlow/design, 1e-4, 1.0);
            const auto request = coolingRequest(thermal, node, node.on, false, true, false, &flows);
            const auto demandFlow = aircon::airflow::computeTargetFlowFromProcessedHeat(
                node, OperationMode::Cooling, std::max(1.0, request.coolingW()), 0.0, minimumW);
            search.tried.push_back(search.ratio);
            if (demandFlow) search.enqueue(*demandFlow/design);
        } else {
            search.tried.push_back(search.ratio);
        }
        if (!std::isfinite(search.best.temperatureExcess) || aircon::operating_point::better(score, search.best)) {
            search.best = score;
            search.bestRatio = search.ratio;
            search.bestOn = node.on;
            search.bestSetpoint = node.current_pre_temp;
            search.bestHumidityHold = node.humidity_overcool_hold;
            search.bestCapacityLimited = node.aircon_control_state == AirconControlState::CapacityLimited;
            search.bestFanTrialDone = humidityFanTrialDone_.count(key) != 0;
        }
        // This is a controller, not an hourly electricity optimizer. Once an
        // alternative satisfies both requirements, stop expanding the grid.
        // A cold start still evaluates the demand-based alternative to full flow.
        if ((solvedAlternative || search.seededFromPrevious) && score.feasible()) {
            search.complete = true;
            logSelection();
            continue;
        }
        // Try the demand-based point first. Expand the bounded search only
        // when none of the solved points meet both room requirements.
        if (search.pending.empty() && !search.expanded && !search.best.feasible()) {
            search.expanded = true;
            for (double fraction : {0.0, 0.25, 0.5, 0.75, 1.0})
                search.enqueue(search.minimumRatio + fraction*(1.0-search.minimumRatio));
        }
        double next = search.bestRatio;
        bool nextOn = search.bestOn;
        double nextSetpoint = search.bestSetpoint;
        bool nextHumidityHold = search.bestHumidityHold;
        bool nextCapacityLimited = search.bestCapacityLimited;
        bool nextFanTrialDone = search.bestFanTrialDone;
        if (!search.pending.empty()) {
            next = search.pending.front();
            search.pending.erase(search.pending.begin());
            nextOn = true;
            // Preserve the achieved room temperature when changing airflow.
            // Resetting an overcooled room straight to the requested (warmer)
            // temperature can demand extreme, unrequested heating at low flow
            // before the non-heating humidistat has a chance to run.
            nextSetpoint = aircon::humidity_control::simultaneous(node)
                ? node.current_requested_pre_temp
                : std::max(room.current_t, node.current_requested_pre_temp-node.max_overcooling);
            nextHumidityHold = !aircon::humidity_control::simultaneous(node);
            nextCapacityLimited = nextSetpoint > node.current_requested_pre_temp+1e-5;
            nextFanTrialDone = false;
        } else {
            search.restoring = true;
            if (std::abs(next-search.ratio) < 1e-6 && nextOn == node.on) {
                search.complete = true;
                logSelection();
                continue;
            }
        }
        std::ostringstream message;
        message << key << " airflow candidate " << search.ratio << "->" << next
            << ", room_T=" << room.current_t << ", room_RH=" << rh
            << ", temperature_excess=" << score.temperatureExcess
            << ", effective_T=" << node.current_pre_temp
            << ", state=" << airconControlStateName(node.aircon_control_state)
            << ", hold=" << node.humidity_overcool_hold << ", demand=" << node.humidity_demand
            << ", fan_trial=" << humidityFanTrialDone_.count(key)
            << ", next_T=" << nextSetpoint << ", next_hold=" << nextHumidityHold
            << ", restore=" << search.restoring;
        writeDomainLog(logs, "空調運転点", message.str());
        captureAirflowCandidate(thermal, ventilation, key, search.ratio);
        search.ratio = next;
        capacityLimitBracket_.clear();
        capacityFlowBoost_.erase(key);
        capacityRecoveryAttempted_.erase(key);
        humidityFloorBlocked_.erase(key);
        humidityFanTrialDone_.erase(key);
        if (nextFanTrialDone) humidityFanTrialDone_.insert(key);
        boundaryIterations_.erase(key);
        for (const auto& other : getAirconKeys()) humiditySearchRebase_.insert(other);
        node.on = nextOn;
        node.aircon_control_state = nextOn ? AirconControlState::SetpointControlled : AirconControlState::Off;
        if (nextOn && nextCapacityLimited)
            node.aircon_control_state = AirconControlState::CapacityLimited;
        node.current_pre_temp = nextSetpoint;
        node.humidity_overcool_hold = nextOn && nextHumidityHold;
        node.humidity_capacity_limited = false;
        node.aircon_min_process_hold = false;
        node.humidity_search_low = node.humidity_search_high = std::numeric_limits<double>::quiet_NaN();
        if (!nextOn) humidityFloorBlocked_.insert(key);
        bool fan = false;
        const double target = nextOn ? next*design : node.vol_zero;
        aircon::airflow::updateDuctCentralFanAffinity(ventilation, node, OperationMode::Cooling,
            key, target, 1e-6, &fan, false, false);
        if (!fan) aircon::airflow::updateDuctCentralCircuitFixedFlows(
            ventilation, node.in_node, key, target, 1e-6);
        return true;
    }
    return false;
}

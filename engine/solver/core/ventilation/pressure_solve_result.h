#pragma once

#include "../../vtsim_solver.h"
#include "core/ventilation/pressure_balance.h"

#include <string>
#include <tuple>

// 圧力ソルバの戻り値。公開 API（VentilationNetwork::solvePressure）は asTuple() で互換維持。
struct PressureSolveResult {
    PressureMap pressures;
    FlowRateMap flows;
    FlowBalanceMap balances;
    bool accepted = false;
    ventilation::BalanceMetrics metrics{};
    // "primary" / "fallback_warmstart" / ""（未収束など）
    std::string method;

    std::tuple<PressureMap, FlowRateMap, FlowBalanceMap> asTuple() const {
        return std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>(pressures, flows, balances);
    }
};

inline PressureSolveResult makePressureSolveResult(
        PressureMap pressures,
        FlowRateMap flows,
        FlowBalanceMap balances,
        bool accepted = false,
        ventilation::BalanceMetrics metrics = {},
        std::string method = {}) {
    PressureSolveResult r;
    r.pressures = std::move(pressures);
    r.flows = std::move(flows);
    r.balances = std::move(balances);
    r.accepted = accepted;
    r.metrics = metrics;
    r.method = std::move(method);
    return r;
}

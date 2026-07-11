#pragma once

#include "../../vtsim_solver.h"
#include "core/ventilation/pressure_balance.h"

#include <string>
#include <tuple>
#include <utility>

// 圧力ソルバの詳細戻り値。公開の 3 変数 structured binding は solvePressures() 側で互換維持。
struct PressureSolveResult {
    PressureMap pressures;
    FlowRateMap flows;
    FlowBalanceMap balances;
    bool accepted = false;
    ventilation::BalanceMetrics metrics{};
    // "primary" / "fallback_warmstart" / ""（未収束など）
    std::string method;

    using TupleType = std::tuple<PressureMap, FlowRateMap, FlowBalanceMap>;

    TupleType asTuple() const& {
        return TupleType{pressures, flows, balances};
    }

    TupleType asTuple() && {
        return TupleType{
            std::move(pressures),
            std::move(flows),
            std::move(balances)};
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

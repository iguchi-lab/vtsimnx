#pragma once

#include <cstdint>
#include <string>
#include <vector>

/** 空調の運転状態（設定温度拘束 vs 能力上限）。 */
enum class AirconControlState : std::uint8_t {
    Off = 0,
    SetpointControlled = 1,
    CapacityLimited = 2,
};

inline const char* airconControlStateName(AirconControlState s) {
    switch (s) {
        case AirconControlState::Off: return "Off";
        case AirconControlState::SetpointControlled: return "SetpointControlled";
        case AirconControlState::CapacityLimited: return "CapacityLimited";
    }
    return "Unknown";
}

/** 外側ループ再計算理由（ビットフラグ）。 */
enum class AirconRecomputeReason : std::uint32_t {
    None = 0,
    OnOffChanged = 1u << 0,
    CapacitySetpointChanged = 1u << 1,
    AirflowChanged = 1u << 2,
    SupplyHumidityChanged = 1u << 3,
};

inline constexpr AirconRecomputeReason operator|(AirconRecomputeReason a, AirconRecomputeReason b) {
    return static_cast<AirconRecomputeReason>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline constexpr AirconRecomputeReason operator&(AirconRecomputeReason a, AirconRecomputeReason b) {
    return static_cast<AirconRecomputeReason>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}
inline AirconRecomputeReason& operator|=(AirconRecomputeReason& a, AirconRecomputeReason b) {
    a = a | b;
    return a;
}
inline constexpr bool hasReason(AirconRecomputeReason flags, AirconRecomputeReason bit) {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(bit)) != 0;
}

/**
 * 1台分の状態提案。外側ループはこれを集約して再計算要否を決める。
 */
struct AirconStateProposal {
    std::string airconKey;
    bool on = false;
    AirconControlState state = AirconControlState::Off;
    double requestedSetpoint = 0.0;   // スケジュール設定温度
    double effectiveSetpoint = 0.0;   // 能力制限などを反映した拘束温度（熱ソルバ fixed-row）
    double processedHeatW = 0.0;
    double maxCapacityW = 0.0;        // 不明時は hasMaxCapacity=false
    bool hasMaxCapacity = false;
    double targetFlowRate = 0.0;
    double supplyHumidity = 0.0;
    AirconRecomputeReason reasons = AirconRecomputeReason::None;
};

/** ノードの現在状態から提案の共通フィールドを埋める。 */
template <typename NodeProps>
inline AirconStateProposal makeAirconStateProposalBase(const std::string& airconKey,
                                                       const NodeProps& node) {
    AirconStateProposal p;
    p.airconKey = airconKey;
    p.on = node.on;
    p.state = node.aircon_control_state;
    p.requestedSetpoint = node.current_requested_pre_temp;
    p.effectiveSetpoint = node.current_pre_temp;
    p.supplyHumidity = node.current_x;
    return p;
}

inline AirconRecomputeReason aggregateProposalReasons(
    const std::vector<AirconStateProposal>& proposals) {
    AirconRecomputeReason all = AirconRecomputeReason::None;
    for (const auto& p : proposals) {
        all |= p.reasons;
    }
    return all;
}

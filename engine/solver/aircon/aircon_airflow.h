#pragma once

#include "aircon/aircon_operation_mode.h"
#include "network/ventilation_network.h"

#include <optional>
#include <string>

namespace aircon::airflow {

bool isDuctCentralModel(const VertexProperties& nodeProps);

bool updateFixedFlowEdgeByNodePair(VentilationNetwork& ventNetwork,
                                   const std::string& fromNode,
                                   const std::string& toNode,
                                   double targetFlowM3s,
                                   double flowTolM3s);

// 還気（in_node -> 空調）と吹出（空調 -> out）の fixed_flow を同じ風量に揃える。
// 向きはビルダーと同じく、吸込は空調へ入り、吹出は空調から出る（どちらも枝方向に +q）。
// in == out のループでも吹出を逆符号にしない。変更があれば true。
bool updateDuctCentralCircuitFixedFlows(VentilationNetwork& ventNetwork,
                                        const std::string& inNode,
                                        const std::string& airconNode,
                                        double targetFlowM3s,
                                        double flowTolM3s);

// subtype=aircon のファン枝があるとき、負荷比 λ=targetFlow/V_dsgn で定格 PQ を相似縮小する。
// q' = λ q_rated、p' = λ² p_rated。実風量は圧力計算の交点であり、枝へ vol は書かない。
// fanPresent が非 null なら、対象ファン枝の有無を返す。曲線が変わったとき true。
bool updateDuctCentralFanAffinity(VentilationNetwork& ventNetwork,
                                  const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  const std::string& airconNode,
                                  double targetFlowM3s,
                                  double ratioTol,
                                  bool* fanPresent = nullptr);

// 風量比に使う基準熱量。計測コイル熱は室温が自由なときだけ。
struct FlowHeatBasis {
    double heatW = 0.0;
    const char* label = "計測処理熱";
    bool exogenous = false;
};

FlowHeatBasis selectFlowHeatBasis(const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  double measuredHeatW,
                                  double controlledRoomTemp);

// 基準熱量から目標風量 [m3/s] を求める。
// - 熱量 <= 0 → 0
// - 0 < 熱量 < Q.min → V_dsgn * Q.min/Q.rtd（最低風量。Q.min が無い機種は線形のまま）
// - それ以上は V_dsgn * clamp(熱量/Q.rtd, 0, 1)
// heldAtMinimum が非 null なら、最低風量で頭打ちしたとき true。
std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW,
                                                         bool* heldAtMinimum = nullptr);

} // namespace aircon::airflow

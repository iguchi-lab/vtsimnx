#pragma once

#include "aircon/aircon_onoff.h"
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
// dampUpdateなら速度比を旧値と目標の中点へ更新。収束判定は緩和前の目標差を使う。
bool updateDuctCentralFanAffinity(VentilationNetwork& ventNetwork,
                                  const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  const std::string& airconNode,
                                  double targetFlowM3s,
                                  double ratioTol,
                                  bool* fanPresent = nullptr,
                                  bool increaseOnly = false,
                                  bool dampUpdate = false);

// 暖房の設定温度拘束を低速ファンで解くと、内側連成が風量補正前に発散しうる。
// 候補の必要負荷に応じてファン速度を増やす。温度候補は呼び出し側で破棄して解き直す。
// 減速・ON/OFF・能力制限・最低能力ホールドの決定は外側制御に残す。
bool increaseHeatingFanForCoupledSolve(VentilationNetwork& ventNetwork,
                                       const Graph& thermalGraph,
                                       std::ostream& logs,
                                       bool logEnabled);

// 風量比に使う基準熱量。計測コイル熱は室温が自由なときだけ。
struct FlowHeatBasis {
    double heatW = 0.0;
    const char* label = "計測処理熱";
    bool exogenous = false;
    bool unmetBoost = false;
};

// 未達で Q_max 風量へ入る温度差、およびその状態を維持する温度差。
// 入る側は 0.5 K を超えた未達。出る側は設定に届くまで Q_max を維持し、
// 19.4°C↔19.6°C で |Qreq|↔Q_max が入れ替わるのを防ぐ。
// 暖房では風量を上げると室温も上がるので、min_process と違いステップ内固定はしない。
constexpr double kUnmetEnterK = 0.5;
constexpr double kUnmetExitK = 0.0;

FlowHeatBasis selectFlowHeatBasis(const VertexProperties& nodeProps,
                                  OperationMode operationMode,
                                  double measuredHeatW,
                                  double controlledRoomTemp,
                                  bool previouslyUnmetBoost = false);

// 基準熱量から目標風量 [m3/s] を求める。
// - 熱量 <= 0 → zeroLoadFlowM3s（既定 0。換気用の零負荷風量）
// - minProcessHeatW > 0 かつ 0 < 熱量 <= minProcessHeatW
//     → V_dsgn * minProcessHeatW/Q.rtd（最低能力時の最低風量）
// - それ以上は V_dsgn * clamp(熱量/Q.rtd, 0, 1)
// ON/OFFホールド履歴は風量を固定しない。正の負荷で下限の前後は連続。
// カタログ Q.min は使わない。heldAtMinimum が非 null なら最低能力風量のとき true。
std::optional<double> computeTargetFlowFromProcessedHeat(const VertexProperties& nodeProps,
                                                         OperationMode operationMode,
                                                         double processedHeatW,
                                                         double zeroLoadFlowM3s = 0.0,
                                                         double minProcessHeatW = 0.0,
                                                         bool* heldAtMinimum = nullptr);

} // namespace aircon::airflow

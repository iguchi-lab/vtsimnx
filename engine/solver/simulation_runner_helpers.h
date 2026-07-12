#pragma once

#include "network/humidity_network.h"
#include "vtsim_solver.h"

#include <vector>

class VentilationNetwork;

namespace simulation {
namespace detail {

void convertDoublesToF32(std::vector<float>& dst, const std::vector<double>& src);
double calculateMaxAbsDiff(const std::vector<double>& oldValues, const std::vector<double>& newValues);
double calculateTemperatureChangeByVertex(const Graph& graph, const std::vector<double>& prevTemps);

bool humidityCouplingActive(const SimulationConstants& constants);
bool needsInnerCoupledIteration(const SimulationConstants& constants);
double couplingPressureTol(const SimulationConstants& constants);
double couplingTemperatureTol(const SimulationConstants& constants);
double couplingHumidityTol(const SimulationConstants& constants);

// 0 なら maxInnerIterations へフォールバック（テストで片方だけ上書きする用途）
std::size_t effectiveMaxCouplingIterations(const SimulationConstants& constants);
std::size_t effectiveMaxAirconControlIterations(const SimulationConstants& constants);

void resetNodeHeatSources(Graph& graph);

// 熱源の内部分離（合成は composeHeatSourcesIntoGraph）
struct SeparatedHeatSources {
    std::vector<double> scheduled;       // 入力スケジュール等
    std::vector<double> airconSensible;  // 空調顕熱
    std::vector<double> humidityLatent;  // 湿気潜熱フィードバック
};

void ensureHeatSourceVectors(SeparatedHeatSources& src, std::size_t nV);
void captureScheduledHeatSources(const Graph& graph, SeparatedHeatSources& src);
void composeHeatSourcesIntoGraph(Graph& graph, const SeparatedHeatSources& src);
double maxAbsLatentHeatChange(const std::vector<double>& prev,
                              const std::vector<double>& curr);

// 同ノード: Q_latent = -rho * V * L * (x_new - x_n) / dt [W]（実験モード）
// V<=0 は 0。raw を計算し、relaxation で prev と混合して humidityLatentOut に書く。
void updateLatentFromHumidityChange(const Graph& graph,
                                    const std::vector<double>& xN,
                                    double dt,
                                    double relaxation,
                                    std::vector<double>& humidityLatentInOut);

// 材料ノードのみ: Q = -L * m_phase, m_phase = -materialPhaseChange（正=蒸発）
void updateLatentFromPhaseChange(const Graph& graph,
                                 const MoistureBalanceTerms& bal,
                                 double relaxation,
                                 std::vector<double>& humidityLatentInOut);

void capturePrevTempsByVertex(const Graph& graph, std::vector<double>& prevTempsByVertex);
void captureXPrevByVertex(const Graph& graph, std::vector<double>& xPrev);
void captureWPrevByVertex(const Graph& graph, std::vector<double>& wPrev);
void capturePrevHumidityByVertex(const Graph& graph, std::vector<double>& prevHumidityByVertex);
void captureHeatSourceByVertex(const Graph& graph, std::vector<double>& heatSourceByVertex);
void restoreHeatSourceByVertex(Graph& graph, const std::vector<double>& heatSourceByVertex);
double calculateHumidityChangeByVertex(const Graph& graph, const std::vector<double>& prevHumidityByVertex);
void relaxHumidityByVertex(Graph& graph,
                           VentilationNetwork& ventNetwork,
                           const std::vector<double>& prevHumidityByVertex,
                           double relaxation);
void restoreXPrevToGraph(Graph& graph, VentilationNetwork& ventNetwork, const std::vector<double>& xPrev);
void restoreWPrevToGraph(Graph& graph, const std::vector<double>& wPrev);

struct CoupledDelta {
    double pressureChange = 0.0;     // [Pa]
    double temperatureChange = 0.0;  // [K]
    double humidityChange = 0.0;     // [kg/kg(DA)]
    double latentHeatChange = 0.0;   // |ΔQ| [W]
    double latentHeatScale = 0.0;    // max(|Qold|,|Qnew|) [W]
};

} // namespace detail
} // namespace simulation


#include "simulation_runner_helpers.h"

#include "network/ventilation_network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include <boost/range/iterator_range.hpp>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace simulation {
namespace detail {
namespace {

// float の指数部が 0xFF（Inf/NaN）なら 0 にする（SIMD 向け）
inline void sanitizeFiniteInplace(std::vector<float>& v) {
    if (v.empty()) return;
#if defined(__AVX2__)
    const __m256i expMask = _mm256_set1_epi32(0x7f800000);
    const __m256i expAll1 = _mm256_set1_epi32(0x7f800000);
    const __m256 zeros = _mm256_setzero_ps();
    size_t i = 0;
    const size_t n = v.size();
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v.data() + i);
        __m256i xi = _mm256_castps_si256(x);
        __m256i exp = _mm256_and_si256(xi, expMask);
        __m256i isInfNaN = _mm256_cmpeq_epi32(exp, expAll1);
        // isInfNaN の lane が 0xFFFFFFFF のところを 0 にする
        __m256 mask = _mm256_castsi256_ps(isInfNaN);
        __m256 cleaned = _mm256_blendv_ps(x, zeros, mask);
        _mm256_storeu_ps(v.data() + i, cleaned);
    }
    for (; i < n; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) v[i] = 0.0f;
    }
#else
    for (float& f : v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) f = 0.0f;
    }
#endif
}

inline int activeCouplingStateCount(const SimulationConstants& constants) {
    int n = 0;
    if (constants.pressureCalc) ++n;
    if (constants.temperatureCalc) ++n;
    if (constants.humidityCalc && constants.moistureCouplingEnabled) ++n;
    return n;
}

} // namespace

void convertDoublesToF32(std::vector<float>& dst, const std::vector<double>& src) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        const double x = src[i];
        if (!std::isfinite(x)) {
            throw std::runtime_error(
                "non-finite value in timestep result (index=" + std::to_string(i) + ")");
        }
        dst[i] = static_cast<float>(x);
    }
}

double calculateMaxAbsDiff(const std::vector<double>& oldValues, const std::vector<double>& newValues) {
    const size_t n = std::min(oldValues.size(), newValues.size());
    double maxChange = 0.0;
    for (size_t i = 0; i < n; ++i) {
        maxChange = std::max(maxChange, std::abs(newValues[i] - oldValues[i]));
    }
    // サイズ不一致は設計上想定しないが、念のため差分を「大きい」とみなす
    if (oldValues.size() != newValues.size()) {
        maxChange = std::max(maxChange, std::numeric_limits<double>::infinity());
    }
    return maxChange;
}

double calculateTemperatureChangeByVertex(const Graph& graph, const std::vector<double>& prevTemps) {
    double maxChange = 0.0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevTemps.size()) continue;
        const double change = std::abs(graph[v].current_t - prevTemps[idx]);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

bool humidityCouplingActive(const SimulationConstants& constants) {
    return constants.humidityCalc && constants.moistureCouplingEnabled;
}

bool needsInnerCoupledIteration(const SimulationConstants& constants) {
    // 連成対象状態量が2つ以上なら、内側反復で収束させる
    return activeCouplingStateCount(constants) >= 2;
}

double couplingPressureTol(const SimulationConstants& constants) {
    return (constants.couplingPressureTolerance > 0.0) ? constants.couplingPressureTolerance
                                                       : constants.convergenceTolerance;
}

double couplingTemperatureTol(const SimulationConstants& constants) {
    return (constants.couplingTemperatureTolerance > 0.0) ? constants.couplingTemperatureTolerance
                                                          : constants.convergenceTolerance;
}

double couplingHumidityTol(const SimulationConstants& constants) {
    return (constants.couplingHumidityTolerance > 0.0) ? constants.couplingHumidityTolerance
                                                       : constants.convergenceTolerance;
}

std::size_t effectiveMaxCouplingIterations(const SimulationConstants& constants) {
    return constants.maxCouplingIterations > 0 ? constants.maxCouplingIterations
                                               : constants.maxInnerIterations;
}

std::size_t effectiveMaxAirconControlIterations(const SimulationConstants& constants) {
    return constants.maxAirconControlIterations > 0 ? constants.maxAirconControlIterations
                                                    : constants.maxInnerIterations;
}

void ensureHeatSourceVectors(SeparatedHeatSources& src, std::size_t nV) {
    auto ensureSize = [nV](std::vector<double>& v) {
        if (v.size() != nV) {
            v.assign(nV, 0.0);
        }
    };
    ensureSize(src.scheduled);
    ensureSize(src.airconSensible);
    ensureSize(src.humidityLatent);
}

void captureScheduledHeatSources(const Graph& graph, SeparatedHeatSources& src) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(graph));
    ensureHeatSourceVectors(src, nV);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        src.scheduled[static_cast<size_t>(v)] = graph[v].heat_source;
    }
}

void composeHeatSourcesIntoGraph(Graph& graph, const SeparatedHeatSources& src) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(graph));
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        double total = 0.0;
        if (i < src.scheduled.size()) total += src.scheduled[i];
        if (i < src.airconSensible.size()) total += src.airconSensible[i];
        if (i < src.humidityLatent.size()) total += src.humidityLatent[i];
        graph[v].heat_source = total;
        (void)nV;
    }
}

double maxAbsLatentHeatChange(const std::vector<double>& prev, const std::vector<double>& curr) {
    return calculateMaxAbsDiff(prev, curr);
}

void updateLatentFromHumidityChange(const Graph& graph,
                                    const std::vector<double>& xN,
                                    double dt,
                                    double relaxation,
                                    std::vector<double>& humidityLatentInOut) {
    if (!(dt > 0.0)) {
        return;
    }
    const size_t nV = static_cast<size_t>(boost::num_vertices(graph));
    if (humidityLatentInOut.size() != nV) {
        humidityLatentInOut.assign(nV, 0.0);
    }
    const std::vector<double> prev = humidityLatentInOut;
    double alpha = std::min(1.0, std::max(0.0, relaxation));
    const double rho = PhysicalConstants::DENSITY_DRY_AIR;
    const double L = PhysicalConstants::LATENT_HEAT_VAPORIZATION;

    // 生値を先に計算し、符号反転（振動）があれば alpha を 0.5 以下へ落とす
    std::vector<double> qRaw(nV, 0.0);
    bool oscillating = false;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        const double V = graph[v].v;
        if (V > 0.0 && i < xN.size()) {
            qRaw[i] = -rho * V * L * (graph[v].current_x - xN[i]) / dt;
        }
        if (prev[i] != 0.0 && qRaw[i] != 0.0 && qRaw[i] * prev[i] < 0.0) {
            oscillating = true;
        }
    }
    if (oscillating) {
        alpha = std::min(alpha, 0.5);
    }

    for (size_t i = 0; i < nV; ++i) {
        humidityLatentInOut[i] = (1.0 - alpha) * prev[i] + alpha * qRaw[i];
    }
}

void updateLatentFromPhaseChange(const Graph& graph,
                                 const MoistureBalanceTerms& bal,
                                 double relaxation,
                                 std::vector<double>& humidityLatentInOut) {
    const size_t nV = static_cast<size_t>(boost::num_vertices(graph));
    if (humidityLatentInOut.size() != nV) {
        humidityLatentInOut.assign(nV, 0.0);
    }
    const std::vector<double> prev = humidityLatentInOut;
    double alpha = std::min(1.0, std::max(0.0, relaxation));
    const double L = PhysicalConstants::LATENT_HEAT_VAPORIZATION;

    std::vector<double> qRaw(nV, 0.0);
    bool oscillating = false;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        // 材料ノードのみ: m_phase = -materialPhaseChange（正=蒸発）, Q = -L * m_phase
        if (graph[v].moisture_capacity > 0.0 && i < bal.materialPhaseChange.size()) {
            const double mPhase = -bal.materialPhaseChange[i];
            qRaw[i] = -L * mPhase;
        }
        if (prev[i] != 0.0 && qRaw[i] != 0.0 && qRaw[i] * prev[i] < 0.0) {
            oscillating = true;
        }
    }
    if (oscillating) {
        alpha = std::min(alpha, 0.5);
    }
    for (size_t i = 0; i < nV; ++i) {
        humidityLatentInOut[i] = (1.0 - alpha) * prev[i] + alpha * qRaw[i];
    }
}

void capturePrevTempsByVertex(const Graph& graph, std::vector<double>& prevTempsByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    prevTempsByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        prevTempsByVertex[static_cast<size_t>(v)] = graph[v].current_t;
    }
}

void captureXPrevByVertex(const Graph& graph, std::vector<double>& xPrev) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    xPrev.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        xPrev[static_cast<size_t>(v)] = graph[v].current_x;
    }
}

void captureWPrevByVertex(const Graph& graph, std::vector<double>& wPrev) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    wPrev.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        wPrev[static_cast<size_t>(v)] = graph[v].current_w;
    }
}

void capturePrevHumidityByVertex(const Graph& graph, std::vector<double>& prevHumidityByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    prevHumidityByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        prevHumidityByVertex[static_cast<size_t>(v)] = graph[v].current_x;
    }
}

void captureHeatSourceByVertex(const Graph& graph, std::vector<double>& heatSourceByVertex) {
    const size_t vCount = static_cast<size_t>(boost::num_vertices(graph));
    heatSourceByVertex.resize(vCount);
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        heatSourceByVertex[static_cast<size_t>(v)] = graph[v].heat_source;
    }
}

void restoreHeatSourceByVertex(Graph& graph, const std::vector<double>& heatSourceByVertex) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < heatSourceByVertex.size()) {
            graph[v].heat_source = heatSourceByVertex[i];
        }
    }
}

double calculateHumidityChangeByVertex(const Graph& graph, const std::vector<double>& prevHumidityByVertex) {
    double maxChange = 0.0;
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevHumidityByVertex.size()) continue;
        if (!graph[v].calc_x) continue;
        const double change = std::abs(graph[v].current_x - prevHumidityByVertex[idx]);
        maxChange = std::max(maxChange, change);
    }
    return maxChange;
}

void relaxHumidityByVertex(Graph& graph,
                           VentilationNetwork& ventNetwork,
                           const std::vector<double>& prevHumidityByVertex,
                           double relaxation) {
    if (!(relaxation > 0.0) || relaxation >= 1.0) return;
    const auto& vKeyToV = ventNetwork.getKeyToVertex();
    auto& vGraph = ventNetwork.getGraph();
    const double alpha = std::min(1.0, std::max(0.0, relaxation));
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t idx = static_cast<size_t>(v);
        if (idx >= prevHumidityByVertex.size()) continue;
        if (!graph[v].calc_x) continue;
        const double mixed = prevHumidityByVertex[idx] + alpha * (graph[v].current_x - prevHumidityByVertex[idx]);
        graph[v].current_x = mixed;
        graph[v].current_w = mixed;
        const auto itV = vKeyToV.find(graph[v].key);
        if (itV != vKeyToV.end()) {
            vGraph[itV->second].current_x = mixed;
        }
    }
}

void restoreXPrevToGraph(Graph& graph, VentilationNetwork& ventNetwork, const std::vector<double>& xPrev) {
    const auto& vKeyToV = ventNetwork.getKeyToVertex();
    auto& vGraph = ventNetwork.getGraph();
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < xPrev.size()) {
            graph[v].current_x = xPrev[i];
            // vent 側にも反映（humidity_solver が vGraph も更新するため）
            const auto itV = vKeyToV.find(graph[v].key);
            if (itV != vKeyToV.end()) {
                vGraph[itV->second].current_x = xPrev[i];
            }
        }
    }
}

void restoreWPrevToGraph(Graph& graph, const std::vector<double>& wPrev) {
    for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
        const size_t i = static_cast<size_t>(v);
        if (i < wPrev.size()) {
            graph[v].current_w = wPrev[i];
        }
    }
}

} // namespace detail
} // namespace simulation


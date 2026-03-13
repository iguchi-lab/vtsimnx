#pragma once

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
};

} // namespace detail
} // namespace simulation


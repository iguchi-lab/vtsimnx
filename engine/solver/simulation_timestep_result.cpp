#include "simulation_timestep_result.h"

#include "aircon/aircon_controller.h"
#include "network/contaminant_network.h"
#include "network/humidity_network.h"
#include "network/thermal_network.h"
#include "network/ventilation_network.h"
#include "simulation_runner_helpers.h"

void buildTimestepResult(const SimulationConstants& constants,
                         VentilationNetwork& ventNetwork,
                         ThermalNetwork& thermalNetwork,
                         HumidityNetwork& humidityNetwork,
                         ContaminantNetwork& contaminantNetwork,
                         AirconController& airconController,
                         const FlowRateMap& flowRates,
                         std::ostream& logs,
                         TimestepResult& timestepResultOut) {
    using simulation::detail::convertDoublesToF32;

    TimestepResult timestepResult;

    if (constants.pressureCalc) {
        convertDoublesToF32(timestepResult.pressure, ventNetwork.collectPressureValues());
    }
    // 換気回路網を構築している場合は風量を出力する（圧力収束計算をしない固定流量のみのときも固定値を出力）
    if (constants.pressureCalc || constants.temperatureCalc || constants.humidityCalc || constants.concentrationCalc) {
        convertDoublesToF32(timestepResult.flowRate, ventNetwork.collectFlowRateValues());
    }

    if (constants.temperatureCalc) {
        convertDoublesToF32(timestepResult.temperature, thermalNetwork.collectTemperatureValues());
        convertDoublesToF32(timestepResult.temperatureCapacity, thermalNetwork.collectTemperatureValuesCapacity());
        convertDoublesToF32(timestepResult.temperatureLayer, thermalNetwork.collectTemperatureValuesLayer());
        convertDoublesToF32(timestepResult.heatRateAdvection, thermalNetwork.collectHeatRateValuesAdvection());
        convertDoublesToF32(timestepResult.heatRateHeatGeneration, thermalNetwork.collectHeatRateValuesHeatGeneration());
        convertDoublesToF32(timestepResult.heatRateSolarGain, thermalNetwork.collectHeatRateValuesSolarGain());
        convertDoublesToF32(timestepResult.heatRateNocturnalLoss, thermalNetwork.collectHeatRateValuesNocturnalLoss());
        convertDoublesToF32(timestepResult.heatRateConvection, thermalNetwork.collectHeatRateValuesConvection());
        convertDoublesToF32(timestepResult.heatRateConduction, thermalNetwork.collectHeatRateValuesConduction());
        convertDoublesToF32(timestepResult.heatRateRadiation, thermalNetwork.collectHeatRateValuesRadiation());
        convertDoublesToF32(timestepResult.heatRateCapacity, thermalNetwork.collectHeatRateValuesCapacity());

        convertDoublesToF32(timestepResult.airconSensibleHeat,
                            airconController.collectAirconDataValues(thermalNetwork, flowRates, "sensibleHeatCapacity"));
        convertDoublesToF32(timestepResult.airconLatentHeat,
                            airconController.collectAirconDataValues(thermalNetwork, flowRates, "latentHeatCapacity"));
        convertDoublesToF32(timestepResult.airconPower,
                            airconController.calculatePowerValues(thermalNetwork, flowRates, logs));
        convertDoublesToF32(timestepResult.airconCOP,
                            airconController.calculateCOPValues(thermalNetwork, flowRates, logs));
    }

    if (constants.humidityCalc) {
        convertDoublesToF32(
            timestepResult.humidityX,
            humidityNetwork.collectOutputValues(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView()));
        convertDoublesToF32(
            timestepResult.humidityFlux,
            ventNetwork.collectHumidityFluxValues());
    }
    if (constants.concentrationCalc) {
        convertDoublesToF32(
            timestepResult.concentrationC,
            contaminantNetwork.collectOutputValues(static_cast<const ThermalNetwork&>(thermalNetwork).nodeStateView()));
        convertDoublesToF32(
            timestepResult.concentrationFlux,
            ventNetwork.collectConcentrationFluxValues());
    }

    timestepResultOut = std::move(timestepResult);
}

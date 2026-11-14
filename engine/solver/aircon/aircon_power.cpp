#include "aircon/aircon_controller.h"
#include "aircon/aircon_latent.h"
#include "aircon/aircon_airflow.h"
#include "aircon/aircon_network_utils.h"
#include "network/thermal_network.h"
#include "archenv/include/archenv.h"
#include "acmodel/duct_central_model.h"
#include "simulation_error.h"
#include "utils/utils.h"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

std::pair<double, double> AirconController::estimatePowerAndCOPForAircon(
    const std::string& airconKey,
    ThermalNetwork& thermalNetwork,
    const VertexProperties& nodeProps,
    const FlowRateMap& flowRates,
    std::ostream& logs,
    bool logDetail) const {
    auto context = prepareRuntimeContext(airconKey, thermalNetwork, nodeProps, flowRates);
    auto* model = getModel(airconKey);
    const auto loads = aircon::latent::estimateLatentProcess(
        context.validData, context.operationMode, context.heatCapacity, context.airFlowRate,
        nodeProps, moistEnthalpyEnabled_);
    if (nodeProps.model == "IDEAL")
        return {nodeProps.on ? loads.reheatCapacity / nodeProps.reheat_cop : 0.0, 0.0};
    if (!model) throw std::runtime_error("初期化済みモデルがありません");
    // No compressor load: account only for intentional ventilation and reheat.
    const double totalHeatW = aircon::latent::totalHeatCapacity(loads);
    if (!nodeProps.on || !(totalHeatW > 0.0)) {
        double fanPowerW = 0.0;
        if (auto* duct = dynamic_cast<acmodel::DuctCentralModel*>(model)) {
            // A stopped fan may still admit passive pressure-driven flow.
            // Only intentional powered ventilation consumes fan electricity.
            if (nodeProps.on || nodeProps.vol_zero > 0.0)
                fanPowerW = 1000.0 * duct->estimateFanPower(
                    modeKey(nodeProps.on ? context.operationMode : aircon::network_utils::fanMode(nodeProps)), context.airFlowRate);
        }
        if (logDetail && logVerbosity_ >= 1) {
            std::ostringstream detail;
            detail << "　　エアコン電力計算: " << airconKey
                   << " [" << modeKey(context.operationMode) << "]"
                   << " 圧縮機停止、送風電力=" << fanPowerW << "W"
                   << " 再熱=" << (nodeProps.on ? loads.reheatCapacity / nodeProps.reheat_cop : 0.0) << "W COP=0";
            writeDomainLog(logs, "空調", detail.str());
        }
        return {fanPowerW + (nodeProps.on ? loads.reheatCapacity / nodeProps.reheat_cop : 0.0), 0.0};
    }
    // 出力・COP 計算はグラフ状態を変更しない（supplyX 適用は外側ループ側）
    acmodel::InputData input =
        aircon::latent::buildAcmodelInput(context.validData,
                                          loads.sensibleHeatCapacity, loads.latentHeatCapacity,
                                          context.airFlowRate);

    const bool heating = isHeating(context.operationMode);
    bool usedFallbackEx = false;
    bool usedFallbackIn = false;
    if (!(input.X_ex > 0.0)) {
        usedFallbackEx = true;
        input.X_ex = heating ? archenv::jis::X_H_EX : archenv::jis::X_C_EX;
    }
    if (!(input.X_in > 0.0)) {
        usedFallbackIn = true;
        input.X_in = heating ? archenv::jis::X_H_IN : archenv::jis::X_C_IN;
    }
    if (logDetail && logVerbosity_ >= 1 && (usedFallbackEx || usedFallbackIn)) {
        std::ostringstream warn;
        warn << "　　[WARN] エアコン湿度入力が不足のためJIS条件で補完: " << airconKey
             << " [" << modeKey(context.operationMode) << "]";
        if (usedFallbackIn) warn << " in_node=" << nodeProps.in_node << " X_in=JIS";
        if (usedFallbackEx) warn << " outside_node=" << nodeProps.outside_node << " X_ex=JIS";
        writeDomainLog(logs, "空調", warn.str());
    }
    if (logDetail && logVerbosity_ >= 1 && loads.saturationLimited) {
        std::ostringstream warn;
        warn << "　　[WARN] 顕熱優先の吹出飽和制約を適用: " << airconKey
             << " RH(model)=" << std::fixed << std::setprecision(2) << loads.unboundedSupplyRhPercent
             << "% -> RH(out)=" << std::fixed << std::setprecision(2) << loads.supplyRhPercent
             << "% (Tout=" << context.validData.airconTemp << "°C, X_out=" << loads.supplyX
             << ", T_coil=" << loads.coilTemp << "°C, X_coil=" << loads.coilX << ")";
        writeDomainLog(logs, "空調", warn.str());
    }
    auto result = model->estimateCOP(modeKey(context.operationMode), input);
    if (logVerbosity_ >= 2) {
        for (const auto& msg : result.logMessages) {
            writeDomainLog(logs, "空調", msg);
        }
    }
    if (!result.valid || !std::isfinite(result.power) || !(result.power > 0.0) ||
        !std::isfinite(result.COP) || result.COP < 0.0) {
        throw std::runtime_error("COP推定に失敗しました: model=" + nodeProps.model +
            ", T_in=" + std::to_string(input.T_in) + ", T_ex=" + std::to_string(input.T_ex) +
            ", Q_W=" + std::to_string(totalHeatW) + ", flow_m3s=" + std::to_string(input.V_inner));
    }
    const double powerW = result.power * 1000.0 + loads.reheatCapacity / nodeProps.reheat_cop;
    const double combinedCOP = powerW > 0.0 ? totalHeatW / powerW : 0.0;
    if (logDetail && logVerbosity_ >= 1) {
        std::ostringstream detail;
        detail << "　　エアコン電力計算: " << airconKey
               << " [" << modeKey(context.operationMode) << "]"
               << " 顕熱=" << std::fixed << std::setprecision(2) << loads.sensibleHeatCapacity << "W"
               << " 潜熱=" << std::fixed << std::setprecision(2) << loads.latentHeatCapacity << "W"
               << " 合計=" << std::fixed << std::setprecision(2) << aircon::latent::totalHeatCapacity(loads) << "W"
               << " 風量=" << context.airFlowRate << "m³/s"
               << " 外気=" << context.validData.outdoorTemp << "°C"
               << " 室内=" << context.validData.indoorTemp << "°C"
               << " COP=" << result.COP
               << " 電力=" << powerW << "W";
        writeDomainLog(logs, "空調", detail.str());
    }
    return {powerW, loads.reheatCapacity > 0.0 ? combinedCOP : result.COP};
}

std::vector<double> AirconController::calculatePowerOrCOPValues(ThermalNetwork& thermalNetwork,
                                                                const FlowRateMap& flowRates,
                                                                std::ostream& logs,
                                                                bool returnPower) const {
    const auto& keys = getAirconKeys();
    std::vector<double> values(keys.size(), 0.0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& airconKey = keys[i];
        const auto& nodeProps = thermalNetwork.getNode(airconKey);
        if (!nodeProps.on && !aircon::airflow::isDuctCentralModel(nodeProps)) {
            continue;
        }
        try {
            const auto pair =
                estimatePowerAndCOPForAircon(airconKey, thermalNetwork, nodeProps, flowRates, logs,
                                             returnPower /* power のみ詳細ログ */);
            values[i] = returnPower ? pair.first : pair.second;
        } catch (const std::exception& e) {
            throw simulation::Error(simulation::ErrorCode::AirconPowerFailed,
                "Aircon power evaluation failed: " + airconKey + " - " + e.what());
        }
    }
    return values;
}

std::vector<double> AirconController::calculatePowerValues(ThermalNetwork& thermalNetwork,
                                                           const FlowRateMap& flowRates,
                                                           std::ostream& logs) const {
    return calculatePowerOrCOPValues(thermalNetwork, flowRates, logs, true);
}

std::vector<double> AirconController::calculateCOPValues(ThermalNetwork& thermalNetwork,
                                                         const FlowRateMap& flowRates,
                                                         std::ostream& logs) const {
    return calculatePowerOrCOPValues(thermalNetwork, flowRates, logs, false);
}


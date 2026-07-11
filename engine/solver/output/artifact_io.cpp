#include "output/artifact_io.h"

#include <chrono>

namespace ArtifactIO {

long long epochMillis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

bool writeJsonToFile(const char* path, const nlohmann::json& j, std::string& err) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    if (!ofs) {
        err = std::string("エラー: 出力ファイルを開けません: ") + path;
        return false;
    }
    ofs << j.dump(2) << "\n";
    ofs.flush();
    if (!ofs) {
        err = std::string("エラー: JSON 書き込みに失敗しました: ") + path;
        return false;
    }
    ofs.close();
    if (ofs.fail()) {
        err = std::string("エラー: JSON クローズに失敗しました: ") + path;
        return false;
    }
    return true;
}

nlohmann::json schemaToJson(long length, long timestepSec, const OutputSchema& s) {
    nlohmann::json j;
    j["length"] = length;
    j["timestep_sec"] = timestepSec;
    j["dtype"] = "f32le";
    j["layout"] = "timestep-major";
    j["series"] = {
        {"vent_pressure", {{"keys", s.pressureKeys}}},
        {"vent_flow_rate", {{"keys", s.flowRateKeys}}},
        {"thermal_temperature", {{"keys", s.temperatureKeys}}},
        {"thermal_temperature_capacity", {{"keys", s.temperatureKeysCapacity}}},
        {"thermal_temperature_layer", {{"keys", s.temperatureKeysLayer}}},
        {"humidity_x", {{"keys", s.humidityKeys}}},
        {"humidity_flux", {{"keys", s.humidityFluxKeys}}},
        {"concentration_c", {{"keys", s.concentrationKeys}}},
        {"concentration_flux", {{"keys", s.concentrationFluxKeys}}},
        {"thermal_heat_rate_advection", {{"keys", s.heatRateKeysAdvection}}},
        {"thermal_heat_rate_heat_generation", {{"keys", s.heatRateKeysHeatGeneration}}},
        {"thermal_heat_rate_solar_gain", {{"keys", s.heatRateKeysSolarGain}}},
        {"thermal_heat_rate_nocturnal_loss", {{"keys", s.heatRateKeysNocturnalLoss}}},
        {"thermal_heat_rate_convection", {{"keys", s.heatRateKeysConvection}}},
        {"thermal_heat_rate_conduction", {{"keys", s.heatRateKeysConduction}}},
        {"thermal_heat_rate_radiation", {{"keys", s.heatRateKeysRadiation}}},
        {"thermal_heat_rate_capacity", {{"keys", s.heatRateKeysCapacity}}},
        {"aircon_sensible_heat", {{"keys", s.airconSensibleHeatKeys}}},
        {"aircon_latent_heat", {{"keys", s.airconLatentHeatKeys}}},
        {"aircon_power", {{"keys", s.airconPowerKeys}}},
        {"aircon_cop", {{"keys", s.airconCOPKeys}}},
    };
    return j;
}

bool writeFloat32ArrayBinary(std::ofstream& ofs,
                             const std::vector<float>& v,
                             size_t expectedSize,
                             std::string& err) {
    if (expectedSize == 0) {
        if (!v.empty()) {
            err = "エラー: schema keys が空なのに結果配列が非空です (size=" +
                  std::to_string(v.size()) + ")";
            return false;
        }
        return true;
    }
    if (v.size() != expectedSize) {
        err = "エラー: schema/result サイズ不一致: expected=" + std::to_string(expectedSize) +
              ", actual=" + std::to_string(v.size());
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(expectedSize * sizeof(float)));
    if (!ofs) {
        err = "エラー: float32 バイナリ書き込みに失敗しました";
        return false;
    }
    return true;
}

} // namespace ArtifactIO

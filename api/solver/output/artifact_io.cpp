#include "output/artifact_io.h"

#include <algorithm>
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
    ofs.close();
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

void writeFloat32ArrayBinary(std::ofstream& ofs, const std::vector<float>& v, size_t expectedSize) {
    if (expectedSize == 0) return;
    const size_t n = std::min(expectedSize, v.size());
    if (n > 0) {
        ofs.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(n * sizeof(float)));
    }
    if (n < expectedSize) {
        // ゼロ埋め（不足分）
        thread_local std::vector<float> zeros;
        const size_t need = expectedSize - n;
        if (zeros.size() < need) zeros.assign(need, 0.0f);
        ofs.write(reinterpret_cast<const char*>(zeros.data()),
                  static_cast<std::streamsize>(need * sizeof(float)));
    }
}

} // namespace ArtifactIO



#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// =============================================================================
// ArtifactIO - 出力（schema.json + float32バイナリ）関連の共通処理
// =============================================================================

namespace ArtifactIO {

// 出力スキーマ（キー配列）: 計算結果は配列で圧縮して出す
struct OutputSchema {
    std::vector<std::string> pressureKeys;
    std::vector<std::string> flowRateKeys;
    std::vector<std::string> temperatureKeys;
    std::vector<std::string> heatRateKeys;

    std::vector<std::string> airconSensibleHeatKeys;
    std::vector<std::string> airconLatentHeatKeys;
    std::vector<std::string> airconPowerKeys;
    std::vector<std::string> airconCOPKeys;

    bool initialized = false;
};

// 現在時刻（epoch ms）
long long epochMillis();

// JSON をファイルに書き出す（失敗時は false, err に理由）
bool writeJsonToFile(const char* path, const nlohmann::json& j, std::string& err);

// schema.json（バイナリ出力のメタ情報）
nlohmann::json schemaToJson(long length, long timestepSec, const OutputSchema& s);

// float32 little-endian の配列をバイナリで書き出す
// 1 timestep = expectedSize 個を固定長で出力（不足分は 0.0f で埋める）
void writeFloat32ArrayBinary(std::ofstream& ofs, const std::vector<float>& v, size_t expectedSize);

} // namespace ArtifactIO



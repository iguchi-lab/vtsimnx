#include "acmodel.h"
#include "criepi_model.h"
#include "rac_model.h"
#include "duct_central_model.h"
#include "latent_evaluate_model.h"
#include <stdexcept>
#include <iostream>

namespace acmodel {

// グローバルログ関数のポインタ
static LogFunction g_logger = nullptr;
static int g_logVerbosity = 1;

void setLogger(LogFunction logger) {
    g_logger = logger;
}

void setLogVerbosity(int level) {
    g_logVerbosity = (level < 0) ? 0 : level;
}

void log(const std::string& message, int level) {
    if (!g_logger) {
        return;
    }
    if (level <= g_logVerbosity) {
        g_logger("　　[acmodel] " + message);
    }
}

std::unique_ptr<AirconSpec> AirconModelFactory::createModel(const std::string& typeStr, const nlohmann::json& spec) {
    if (typeStr == "CRIEPI") {
        return std::make_unique<CRIEPIModel>(spec);
    } else if (typeStr == "RAC") {
        return std::make_unique<RACModel>(spec);
    } else if (typeStr == "DUCT_CENTRAL") {
        return std::make_unique<DuctCentralModel>(spec);
    } else if (typeStr == "LATENT_EVALUATE") {
        return std::make_unique<LatentEvaluateModel>(spec);
    } else {
        throw std::runtime_error("エアコンモデルのタイプが不明です");
    }
}

AirconType AirconModelFactory::getTypeFromString(const std::string& typeStr) {
    if (typeStr == "CRIEPI") return AirconType::CRIEPI;
    if (typeStr == "RAC") return AirconType::RAC;
    if (typeStr == "DUCT_CENTRAL") return AirconType::DUCT_CENTRAL;
    if (typeStr == "LATENT_EVALUATE") return AirconType::LATENT_EVALUATE;
    throw std::runtime_error("エアコンモデルのタイプが不明です");
}

void initialize() {
    // モジュールの初期化処理
    // 必要に応じて追加
}

void finalize() {
    // モジュールの終了処理
    // 必要に応じて追加
}

} // namespace acmodel 
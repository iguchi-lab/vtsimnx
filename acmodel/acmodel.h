#pragma once

#include "aircon_spec.h"

#include <memory>
#include <string>
#include <functional>

namespace acmodel {

/**
 * @brief ログ関数の型定義
 */
using LogFunction = std::function<void(const std::string&)>;

/**
 * @brief ログ関数を設定する
 * @param logger ログ関数
 */
void setLogger(LogFunction logger);

/**
 * @brief acmodel内部の詳細ログを出力する最低レベルを設定
 * @param level 0:出力なし, 1:標準, 2:詳細, 3:デバッグ
 */
void setLogVerbosity(int level);

/**
 * @brief ログを出力する
 * @param message ログメッセージ
 * @param level ログ表示に必要なレベル(デフォルト:詳細=2)
 */
void log(const std::string& message, int level = 2);

/**
 * @brief エアコンモデルのタイプ
 */
enum class AirconType {
    CRIEPI,         // CRIEPIモデル
    RAC,            // ルームエアコンディショナーモデル
    DUCT_CENTRAL,   // ダクト式セントラルモデル（未実装）
    LATENT_EVALUATE // 潜熱評価モデル（未実装）
};

/**
 * @brief エアコンモデルのファクトリークラス
 */
class AirconModelFactory {
public:
    /**
     * @brief エアコンモデルを作成する
     * @param type モデルタイプ
     * @param spec エアコン仕様のJSONオブジェクト
     * @return 作成されたエアコンモデルのユニークポインタ
     */
    static std::unique_ptr<AirconSpec> createModel(const std::string& typeStr, const nlohmann::json& spec);

    /**
     * @brief 文字列からエアコンタイプを取得する
     * @param typeStr タイプ文字列 ("CRIEPI", "RAC", "DUCT_CENTRAL", "LATENT_EVALUATE")
     * @return エアコンタイプ
     */
    static AirconType getTypeFromString(const std::string& typeStr);
};

/**
 * @brief ACモジュールの初期化を行う
 */
void initialize();

/**
 * @brief ACモジュールの終了処理を行う
 */
void finalize();

} // namespace acmodel 
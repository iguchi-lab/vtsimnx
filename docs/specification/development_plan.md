# 仕様書の整備計画

[全体目次](README.md)に戻る。

## 1. 作成順序

| 段階 | 対象 | 成果物・完了条件 |
|---|---|---|
| 1 | 第一～五章 | 共通定義、Python API 対応、raw JSON 項目表、変換表、計算用 JSON 項目表。最小例で入力から生成結果まで追跡できる |
| 2 | 第六章 | 計算順、状態保存・更新、収束判定、失敗時処理を確定 |
| 3 | 第七～十章 | 換気・熱・湿気・濃度の一般式、離散式、境界、例外、既知解との照合 |
| 4 | 第十一・十二章 | 設備制御から処理熱量・COP・電力までを接続し、算定範囲と積算方法を明示 |
| 5 | 第十三・十四章、付録 | 実行・出力仕様、検証表、通し例、版対応、改訂履歴 |

出力項目表と仕様確認表は各章の執筆時から記入する。

## 2. 既存資料・実装との対応

以下は実装参照の入口である。全章の本文初稿を作成し、主要処理を対象 commit と照合した。全フィールド・全係数・全境界条件の照合完了を示す表ではない。

| 章 | 既存資料 | 実装の入口 | 重点確認事項 |
|---|---|---|---|
| 一 | [単位](../units.md)、[公開 API](../public_api.md) | [units.py](../../vtsimnx/units.py) | 用語、符号、時刻の統一 |
| 二 | [スケジュール](../schedule_usage.md)、[環境計算](../archenv_comfort_nocturnal_wind_usage.md)、[日射](../solar_usage.md) | [Python モジュール](../../vtsimnx)、[run_calc](../../vtsimnx/run_calc/run_calc.py) | 全公開モジュールの棚卸しと raw JSON との対応 |
| 三 | [ビルダー入力](../../engine/docs/builder_json.md) | [parsers](../../engine/app/builder/parsers.py)、[validate](../../engine/app/builder/validate.py) | 必須性、既定値、範囲、時系列条件 |
| 四 | [ビルダー入力](../../engine/docs/builder_json.md) | [builder](../../engine/app/builder) | 正確な処理順、生成・削除・補正、変換前後の対応 |
| 五 | [計算概略](../../engine/docs/simulation_overview.md) | [schemas](../../engine/app/schemas)、[C++ parser](../../engine/solver/parser) | 計算用 JSON の独立した項目表と受入条件 |
| 六 | [計算ループ](../../engine/docs/simulation_loops.md) | [連成制御](../../engine/solver/simulation_coupling_control.cpp) | 時間の二重進行防止、収束、状態確定 |
| 七 | [物理基礎](../../engine/docs/theory_basics.md) | [換気計算](../../engine/solver/core/ventilation) | 流路別の式、圧力拘束、逆流、解法 |
| 八 | [RC](../../engine/docs/thermal_rc.md)、[応答係数](../../engine/docs/thermal_response_factor.md) | [熱計算](../../engine/solver/core/thermal) | 離散式、行列、係数生成、履歴、近似 |
| 九 | [湿気回路網](../../engine/docs/moisture_network_phase1.md) | [湿気計算](../../engine/solver/core/humidity) | 水分の基準、吸放湿・結露の対応範囲、潜熱連成 |
| 十 | [物理基礎](../../engine/docs/theory_basics.md) | [濃度ソルバ](../../engine/solver/transport/concentration_solver.cpp) | 独立した濃度仕様、発生・沈着・除去の単位、離散式 |
| 十一 | [空調制御](../../engine/docs/aircon_control_overview.md)、[熱交換換気](../heat_recovery_vent.md) | [空調](../../engine/solver/aircon)、[熱交換換気](../../engine/solver/hrv) | 制御対象、処理熱、湿度、四ポート、反復条件 |
| 十二 | [電力モデル](../../engine/docs/acmodel_overview.md) | [acmodel](../../engine/acmodel) | モデル別の式、零負荷、ファン算入、集計の実装有無 |
| 十三 | [HTTP API](../../engine/docs/api_reference.md)、[ログ](../../engine/docs/solver_logging.md) | [成果物出力](../../engine/solver/output/artifact_io.cpp)、[取得](../../vtsimnx/artifacts) | 系列別の単位、時刻、空系列、失敗時 |
| 十四 | [検証方針](../validation_strategy.md)、[仕様対応表](../../engine/docs/specification_traceability.md) | [Python テスト](../../engine/tests_py)、[C++ テスト](../../engine/solver/tests_cpp) | 仕様単位の検証範囲、許容差、未検証条件 |

## 3. 各章の進め方

1. 対象版を固定し、節と仕様 ID を割り当てる。
2. 関連資料・実装・テストを読み、入出力・処理・例外を棚卸しする。
3. 対象 commit の実装に合わせて本文を作成する。既存記述との差異は付録に記録する。
4. 最小例で変換又は計算を照合し、実行と読解のみの確認を区別する。
5. 他章との重複・単位・符号・参照を確認し、未解決事項を管理する。
6. 確定した節について既存仕様との関係及び参照先を更新する。

既存の技術情報・技術解説書への転記や同期は、この整備計画の作業に含めない。

## 4. 現時点の整備状況

| 対象 | 状態 |
|---|---|
| 全体目次 | 章・節の構成案を作成 |
| 共通記述様式 | 初稿を作成 |
| 既存資料・実装の対応 | 参照先を整理。詳細照合は各章の執筆時に実施 |
| 第一～十四章の本文、付録 | 本文初稿0.2を作成。詳細化の残作業は第十四章 |
| 数式・変換例の実行検証 | 実装呼出しと算術例を分けて確認。結果は第十四章・examples参照 |

## 5. 改訂履歴

| 日付 | 文書版 | 内容 |
|---|---|---|
| 2026-09-17 | 構成案 0.1 | 独立した仕様書として目次・執筆要領・整備計画を作成 |
| 2026-09-17 | 本文初稿 0.2 | 14章・付録・再現例を追加。実装を基準に記述する方針を反映 |

# docs index（利用者向け）

`vtsimnx` の Python クライアントで入力を組み立て、`vt.run_calc` で計算するためのガイドです。
engine 実装の厳密仕様は [`../engine/docs/README.md`](../engine/docs/README.md) を参照してください。

## 読み順（最短）

1. [`builder_input_quickstart.md`](builder_input_quickstart.md) — `input_data` の最小形
2. [`node_branch_schema.md`](node_branch_schema.md) — nodes / branches 早見表
3. [`../examples/README.md`](../examples/README.md) — 動くサンプル

入力単位と前処理の換算は [`units.md`](units.md)（コード: `vtsimnx.units` と各前処理）を参照してください。

## 機能ガイド

| ガイド | 内容 |
|---|---|
| [`solar_usage.md`](solar_usage.md) | 日射取得 |
| [`surface_usage.md`](surface_usage.md) | `surfaces` の組み立て |
| [`response_factor_method.md`](response_factor_method.md) | 応答係数法・CTFの式、RCとの関係、履歴と制約 |
| [`aircon_humidity_control.md`](aircon_humidity_control.md) | 空調湿度境界、理想除湿pre_rh、再計算閾値と制約 |
| [`schedule_usage.md`](schedule_usage.md) | 8760 スケジュール |
| [`archenv_comfort_nocturnal_wind_usage.md`](archenv_comfort_nocturnal_wind_usage.md) | 風圧・夜間放射・地盤・PMV/PPD |
| [`vs_simheat_example.md`](vs_simheat_example.md) | SimHeat 比較ケースの入力フロー |
| [`building_environment_engineering_basics.md`](building_environment_engineering_basics.md) | 建築環境の背景（利用者向け） |

## 方針・契約

| 文書 | 内容 |
|---|---|
| [`public_api.md`](public_api.md) | stable / experimental / deprecated |
| [`units.md`](units.md) | 単位系の正本 |
| [`validation_strategy.md`](validation_strategy.md) | 検証ピラミッドと保証範囲 |
| [`release_policy.md`](release_policy.md) | バージョン / tag / リリース運用 |

## 実装仕様が必要なとき

利用者ガイドで足りない場合のみ参照してください（正本は engine 側）。

| 用途 | 正本 |
|---|---|
| HTTP API 契約 | [`../engine/docs/api_reference.md`](../engine/docs/api_reference.md) |
| builder 入力 JSON（厳密） | [`../engine/docs/builder_json.md`](../engine/docs/builder_json.md) |
| 計算フロー全体 | [`../engine/docs/simulation_overview.md`](../engine/docs/simulation_overview.md) |
| 空調制御（遠隔 set＝未設置室の温度制御を含む） | [`../engine/docs/aircon_control_overview.md`](../engine/docs/aircon_control_overview.md) |
| 湿気 Phase1 | [`../engine/docs/moisture_network_phase1.md`](../engine/docs/moisture_network_phase1.md) |

## 技術文書と点検記録

- 現行技術文書v3.2：[PDF](vtsimnx_calculation_model_and_implementation_flow_ja_v3.2.pdf) / [DOCX](vtsimnx_calculation_model_and_implementation_flow_ja_v3.2.docx)。2026年9月改訂。応答係数法は6.7節。
- [v1.7.4差分点検記録](documentation_review_2026-09_v174.md)：今回の改訂内容、確認結果、空調湿気処理の制約。
- [初回文書点検記録](documentation_review_2026-09.md)：基礎式・応答係数法の照合、数値確認、コード側の未解決事項。
- 文書v3.2はパッケージv1.7.4（commit `9a5c379`）の実装を説明します。空調湿度境界と理想除湿の改訂を反映済みです。文書とソフトウェアの版番号は別です。
- v2.8・v3.1は旧版資料です。現行の説明と異なる場合は、対象commitを明記したv3.2改訂版と各ガイドを参照してください。

## 関連入口

- リポジトリ入口: [`../README.md`](../README.md)
- engine 運用: [`../engine/README.md`](../engine/README.md)

# ドキュメント案内

文書は役割ごとに分かれています。上の層から読んでください。下の層は契約と根拠であり、最初の説明ではありません。

```text
1. 使い方      docs/          入力の意味、単位、機能ガイド、実行例
2. 実装契約    engine/docs/   フィールド正本、HTTP API、計算ループ
3. 規範仕様    docs/specification/  章立ての仕様本文（初稿）
4. 解説・記録  PDF / 点検メモ  モデル解説と、当時の確認記録
```

現行ソフトウェアの版は `pyproject.toml`（v1.8.1）です。解説書 PDF の版番号は別です。食い違う場合は、実装契約（`engine/docs/`）とコードを優先します。

## 何をしたいか

| 目的 | 読む場所 |
|---|---|
| 入力 JSON の意味を理解し、組み立てる | [入力JSONの組み立て](ai_input_guide.md) |
| Python から最小計算を動かす | [クイックスタート](builder_input_quickstart.md) → [examples](../examples/README.md) |
| キーと単位を確認する | [ノード／枝の早見表](node_branch_schema.md) / [単位](units.md) |
| 壁・空調・換気などを足す | 下の機能ガイド |
| 項目の定義を厳密に確認する | [builder 入力正本](../engine/docs/builder_json.md) |
| HTTP の要求と応答を確認する | [API 契約](../engine/docs/api_reference.md) |
| 式と計算手順の規範を読む | [仕様書](specification/README.md) |
| モデルの解説を通読する | [技術解説書 v3.3（PDF）](vtsimnx_calculation_model_and_implementation_flow_ja_v3.3.pdf) |

## 最短の読み順

1. [入力JSONの組み立て](ai_input_guide.md) — 室・枝・単位・検査
2. [クイックスタート](builder_input_quickstart.md) — `vt.run_calc` の最小コード
3. [ノード／枝の早見表](node_branch_schema.md) — キー一覧
4. 必要になった機能だけ、下のガイドへ

機械可読な型は `GET /openapi.json` の `RawSimConfig` です。本計算の前に `POST /validate` と `POST /build` で直せます。

## 機能ガイド

必要なものだけ読んでください。

| ガイド | 内容 |
|---|---|
| [`solar_usage.md`](solar_usage.md) | 日射取得 |
| [`surface_usage.md`](surface_usage.md) | `surfaces` の組み立て |
| [`response_factor_method.md`](response_factor_method.md) | 応答係数法・CTF |
| [`aircon_model_and_control.md`](aircon_model_and_control.md) | 空調の接続・制御・潜熱・電力 |
| [`aircon_humidity_control.md`](aircon_humidity_control.md) | 潜熱処理と `pre_rh` |
| [`duct_central_airflow_control.md`](duct_central_airflow_control.md) | 全館空調の風量と最低能力 |
| [`heat_recovery_vent.md`](heat_recovery_vent.md) | 熱交換換気 |
| [`schedule_usage.md`](schedule_usage.md) | 8760 スケジュール |
| [`archenv_comfort_nocturnal_wind_usage.md`](archenv_comfort_nocturnal_wind_usage.md) | 風圧・夜間放射・地盤・PMV |
| [`vs_simheat_example.md`](vs_simheat_example.md) | SimHeat 比較ケース |
| [`building_environment_engineering_basics.md`](building_environment_engineering_basics.md) | 建築環境の背景 |

## 方針

| 文書 | 内容 |
|---|---|
| [`public_api.md`](public_api.md) | 公開 Python API の安定性 |
| [`units.md`](units.md) | 単位の正本 |
| [`validation_strategy.md`](validation_strategy.md) | 検証の範囲と限界 |
| [`release_policy.md`](release_policy.md) | バージョンとリリース |

## 実装契約（必要なときだけ）

使い方のガイドで足りない項目・挙動の正本です。最初の読み物ではありません。

| 用途 | 正本 |
|---|---|
| HTTP API | [`../engine/docs/api_reference.md`](../engine/docs/api_reference.md) |
| builder 入力の全フィールド | [`../engine/docs/builder_json.md`](../engine/docs/builder_json.md) |
| 計算の順序 | [`../engine/docs/simulation_overview.md`](../engine/docs/simulation_overview.md) |
| 空調制御 | [`../engine/docs/aircon_control_overview.md`](../engine/docs/aircon_control_overview.md) |
| 湿気回路 | [`../engine/docs/moisture_network_phase1.md`](../engine/docs/moisture_network_phase1.md) |
| 実装仕様の索引 | [`../engine/docs/README.md`](../engine/docs/README.md) |

## 規範仕様・解説書・点検記録

- 仕様本文: [`specification/README.md`](specification/README.md)。章立ての契約であり、チュートリアルではありません。
- 技術解説書 v3.3: [PDF](vtsimnx_calculation_model_and_implementation_flow_ja_v3.3.pdf) / [DOCX](vtsimnx_calculation_model_and_implementation_flow_ja_v3.3.docx)。パッケージ v1.8.0 時点の説明です。文書版とソフトウェア版は別です。
- 当時の確認メモ: [v1.8.0](documentation_review_2026-09_v180.md) / [v1.7.5](documentation_review_2026-09_v175.md) / [空調章](documentation_review_2026-09_aircon_refinement.md) / [v1.7.4](documentation_review_2026-09_v174.md) / [初回](documentation_review_2026-09.md)。現行と異なる場合は、対象 commit を明記した解説と各ガイドを優先します。

## 関連

- リポジトリ入口: [`../README.md`](../README.md)
- engine の起動: [`../engine/README.md`](../engine/README.md) / [`../engine/RUN_FASTAPI.md`](../engine/RUN_FASTAPI.md)

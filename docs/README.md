# docs index

このディレクトリは、`vtsimnx` 利用者が  
**`vtsimnx/vtsimnx` の API（主に `vt.run_calc`）で入力を組み立てるための実務ガイド**をまとめています。

engine 側の実装詳細は参照先として残しつつ、ここでは「利用者が何をどう書くか」を優先して説明します。

## まず読む

- `builder_input_quickstart.md`
  - `vt.run_calc` に渡す `input_data` の最小構成と、JSONの書き方
- `../examples/README.md`
  - 実際に動かせるサンプルコード（`run_calc_minimal.py`, `vs_simheat_r15.py` と気象データ）
- `building_environment_engineering_basics.md`
  - 熱収支、日射、換気、湿気、快適性の背景
- `node_branch_schema.md`
  - `input_data` の `nodes` / `ventilation_branches` / `thermal_branches` 早見表

## 機能別ガイド（ライブラリ利用）

- `solar_usage.md`
  - `solar_gain_by_angles`, `solar_gain_by_angles_with_shade`
- `archenv_comfort_nocturnal_wind_usage.md`
  - 風圧、夜間放射、地盤温度、PMV/PPD
- `schedule_usage.md`
  - `vtsimnx.schedule` の8760スケジュール設計
- `surface_usage.md`
  - `surfaces` を組み立てるための実務的な考え方
- `vs_simheat_example.md`
  - SimHeat比較ケースでの入力構築フロー

## 信頼性と運用

- `validation_strategy.md`
  - 検証ピラミッド、保証範囲、未保証範囲
- `release_policy.md`
  - version/tag/docs/examples の対応ルール

## APIドキュメントへの境界

以下は実装詳細・厳密仕様を確認したいときの参照先です。

- 実装ドキュメント入口: `../engine/docs/README.md`
- APIエンドポイント契約: `../engine/docs/api_reference.md`
- builder入力JSON仕様: `../engine/docs/builder_json.md`
- solver/aircon挙動の詳細: `../engine/docs/simulation_overview.md`, `../engine/docs/aircon_control_overview.md`

## 関連入口

- リポジトリ入口: `../README.md`
- API運用入口: `../engine/README.md`

## 運用メモ（内部向け）

- `internal/repo_metadata.md`
- `internal/legacy_repo_retirement_checklist.md`


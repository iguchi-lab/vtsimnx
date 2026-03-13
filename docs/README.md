# core docs index

このディレクトリは `vtsimnx` ライブラリ利用者向けの解説をまとめています。  
APIの厳密仕様・運用手順は `engine/docs/` 側を正本として参照してください。

## まず読む

- `building_environment_engineering_basics.md`
  - 熱収支、日射、換気、湿気、快適性の背景
- `node_branch_schema.md`
  - ノード/ブランチ入力の利用者向け整理（早見表）

## 機能別ガイド（core側）

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

## APIドキュメントへの境界

以下は `engine/docs/` を正本とします（core側は概要・導線のみ）。

- APIエンドポイント契約: `../engine/docs/api_reference.md`
- builder入力JSON仕様: `../engine/docs/builder_json.md`
- solver/aircon挙動の詳細: `../engine/docs/simulation_overview.md`, `../engine/docs/aircon_control_overview.md`

## 関連入口

- リポジトリ入口: `../README.md`
- API運用入口: `../engine/README.md`


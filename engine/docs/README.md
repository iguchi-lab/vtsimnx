# engine docs index

このディレクトリは、`engine/` 実装の仕様・設計メモ（API/builder/solver/aircon）の正本です。  
`vt.run_calc` の入力を作る利用者向けガイドは `../../docs/README.md` を参照してください。

## 主要ドキュメント

- `api_reference.md`: APIエンドポイント契約
- `builder_json.md`: builder入力JSON（raw_config）の作り方
- `simulation_overview.md`: シミュレーション全体の概略（builder→solver）
- `aircon_control_overview.md`: エアコン制御ロジックの概要
- `acmodel_overview.md`: エアコンモデル（CRIEPI/RAC/DUCT_CENTRAL など）
- `aircon_spec_reference.md`: `ac_spec` の必須キー・モデル別仕様
- `thermal_rc.md`: 壁モデル（RC法）
- `thermal_response_factor.md`: 壁モデル（応答係数法/CTF）
- `theory_basics.md`, `physics_math_notes.md`: 理論・数理メモ
- `constants_and_spec.md`, `check_outer_surface_colder_than_air.md`: 補助仕様・検証メモ

## 読み方ガイド（目的別）

- 初めて全体像を把握する: `theory_basics.md` → `simulation_overview.md` → `builder_json.md`
- 入力JSON仕様を深掘りする: `builder_json.md` → `thermal_rc.md` / `thermal_response_factor.md`
- 空調の挙動を確認する: `aircon_control_overview.md` → `acmodel_overview.md` → `aircon_spec_reference.md`
- 実装寄りに追う: `simulation_overview.md` → `aircon_control_overview.md` → `acmodel_overview.md`

## 導線

- 利用者向けドキュメント: `../../docs/README.md`
- リポジトリ入口: `../../README.md`
- API運用入口: `../README.md`

## 低負荷でビルド/テストしたいとき（メモ）

CPU/RAMを抑えたい場合は並列数を下げます。

- build: `cmake --build build-solver -j1`（または `-j2`）
- test: `ctest --test-dir build-solver -j1 --output-on-failure`
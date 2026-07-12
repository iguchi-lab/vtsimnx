# engine docs index（実装仕様）

`engine/`（FastAPI / builder / C++ solver）の仕様正本です。  
Python クライアント利用者向けガイドは [`../../docs/README.md`](../../docs/README.md) を先に読んでください。

> このディレクトリ内の相対パス `docs/foo.md` は **本ディレクトリ**（`engine/docs/`）を指します。  
> リポジトリ直下の利用者 doc は `../../docs/foo.md` と書いてください。

図（mermaid）を多用した入口は [`simulation_loops.md`](simulation_loops.md) です。他文書も同形式で概要図を入れています。

## まず読む

| 順序 | 文書 | 内容 |
|---|---|---|
| 1 | [`theory_basics.md`](theory_basics.md) | 実装につながる物理の全体像 |
| 2 | [`simulation_overview.md`](simulation_overview.md) | builder → solver の計算順 |
| 3 | [`simulation_loops.md`](simulation_loops.md) | 外側/内側/空調ループの図解 |
| 4 | [`builder_json.md`](builder_json.md) | raw_config の厳密仕様（入力正本） |

## API / 入力 / 物理

| 文書 | 役割 |
|---|---|
| [`api_reference.md`](api_reference.md) | HTTP エンドポイント契約 |
| [`builder_json.md`](builder_json.md) | builder 入力 JSON 正本 |
| [`simulation_overview.md`](simulation_overview.md) | 連成・タイムステップ概略 |
| [`simulation_loops.md`](simulation_loops.md) | 計算ループ構成（図多用） |
| [`moisture_network_phase1.md`](moisture_network_phase1.md) | 湿気回路網 Phase1 |
| [`physics_math_notes.md`](physics_math_notes.md) | 符号・単位・離散化の注意 |
| [`solver_logging.md`](solver_logging.md) | solver.log の用語・タグ・重大度規約 |
| [`constants_and_spec.md`](constants_and_spec.md) | 定数・材料テーブル対応 |
| [`thermal_rc.md`](thermal_rc.md) | 壁モデル（RC） |
| [`thermal_response_factor.md`](thermal_response_factor.md) | 壁モデル（応答係数/CTF） |

単位の利用者向け正本は [`../../docs/units.md`](../../docs/units.md) です。

## 空調

特徴: **エアコンの吸込・吹出空間と、温度制御対象（`set`）を分離できる**（遠隔 set）。階間空調などで LDK など未設置室を制御する構成が中核機能です。詳細は [`aircon_control_overview.md`](aircon_control_overview.md) 冒頭。

| 文書 | 役割 |
|---|---|
| [`aircon_control_overview.md`](aircon_control_overview.md) | solver 側制御ロジック（遠隔 set 含む） |
| [`acmodel_overview.md`](acmodel_overview.md) | COP / 電力モデル |
| [`aircon_spec_reference.md`](aircon_spec_reference.md) | `ac_spec` キー一覧 |
| [`duct_central_model_validation.md`](duct_central_model_validation.md) | DUCT_CENTRAL 検証観点 |

## テスト

| 文書 | 役割 |
|---|---|
| [`cpp_test_catalog.md`](cpp_test_catalog.md) | C++ テスト観点 |
| [`../tests_py/README.md`](../tests_py/README.md) | Python テスト |
| [`../solver/tests_cpp/README.md`](../solver/tests_cpp/README.md) | C++ テスト最小 README |

## 内部作業メモ

公開仕様ではありません。[`internal/`](internal/README.md) を参照してください。

## 導線

- 利用者向け: [`../../docs/README.md`](../../docs/README.md)
- リポジトリ入口: [`../../README.md`](../../README.md)
- 起動・運用: [`../RUN_FASTAPI.md`](../RUN_FASTAPI.md)
- 開発参加: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)

## 低負荷ビルド（メモ）

- build: `cmake --build build-solver -j1`（cwd: `engine/`）
- test: `ctest --test-dir build-solver -j1 --output-on-failure`

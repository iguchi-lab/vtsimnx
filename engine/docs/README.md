# engine 実装契約

ここは FastAPI / builder / C++ solver の**契約と内部説明**です。入力の意味や使い方は先に [`../../docs/README.md`](../../docs/README.md) を読んでください。

このディレクトリ内の相対パス `docs/foo.md` は **本ディレクトリ**（`engine/docs/`）を指します。利用者向け文書は `../../docs/foo.md` です。

## まず読む（実装を追うとき）

| 順序 | 文書 | 内容 |
|---|---|---|
| 1 | [`builder_json.md`](builder_json.md) | raw 入力のフィールド正本 |
| 2 | [`api_reference.md`](api_reference.md) | HTTP（`/validate`・`/build`・`/runs`） |
| 3 | [`simulation_overview.md`](simulation_overview.md) | builder → solver の計算順 |
| 4 | [`simulation_loops.md`](simulation_loops.md) | 外側／内側／空調ループの図 |

物理の背景から入る場合は [`theory_basics.md`](theory_basics.md) です。入力の書き方ではありません。

単位の正本は利用者向けの [`../../docs/units.md`](../../docs/units.md) です。

## 入力と API

| 文書 | 役割 |
|---|---|
| [`builder_json.md`](builder_json.md) | builder が受ける JSON の正本 |
| [`api_reference.md`](api_reference.md) | エンドポイント契約 |
| [`solver_logging.md`](solver_logging.md) | solver.log の用語 |
| [`constants_and_spec.md`](constants_and_spec.md) | 定数・材料テーブル |

## 熱・湿気

| 文書 | 役割 |
|---|---|
| [`thermal_rc.md`](thermal_rc.md) | 壁（RC） |
| [`thermal_response_factor.md`](thermal_response_factor.md) | 壁（応答係数／CTF） |
| [`moisture_network_phase1.md`](moisture_network_phase1.md) | 湿気回路網 Phase1 |
| [`physics_math_notes.md`](physics_math_notes.md) | 符号・離散化の注意 |

## 空調・熱交換換気

エアコンの吸込・吹出と、温度制御対象（`set`）は分けられます（遠隔 set）。詳細は [`aircon_control_overview.md`](aircon_control_overview.md) 冒頭。

| 文書 | 役割 |
|---|---|
| [`aircon_control_principles.md`](aircon_control_principles.md) | 制御の考え方 |
| [`aircon_control_overview.md`](aircon_control_overview.md) | solver 側の制御 |
| [`acmodel_overview.md`](acmodel_overview.md) | COP / 電力 |
| [`aircon_spec_reference.md`](aircon_spec_reference.md) | `ac_spec` キー |
| [`duct_central_model_validation.md`](duct_central_model_validation.md) | DUCT_CENTRAL の検証観点 |
| [`builder_json.md`](builder_json.md) §11.0 | `heat_recovery_vent` 入力 |
| [`../../docs/heat_recovery_vent.md`](../../docs/heat_recovery_vent.md) | 熱交換換気の利用者ガイド |

## 仕様の保守

[仕様・実装・テスト対応表](specification_traceability.md) に、参照先、テスト範囲、文書の直し方をまとめています。仕様、現行の実装説明、制約、将来案を混ぜないでください。

## テスト・内部メモ

| 文書 | 役割 |
|---|---|
| [`cpp_test_catalog.md`](cpp_test_catalog.md) | C++ テスト観点 |
| [`../tests_py/README.md`](../tests_py/README.md) | Python テスト |
| [`internal/`](internal/README.md) | 公開しない作業メモ |

起動: [`../RUN_FASTAPI.md`](../RUN_FASTAPI.md)　開発: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)　リポジトリ入口: [`../../README.md`](../../README.md)

低負荷ビルドの目安: `cmake --build build-solver -j1`（cwd: `engine/`）、`ctest --test-dir build-solver -j1 --output-on-failure`。

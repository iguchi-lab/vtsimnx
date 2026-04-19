# Refactoring Backlog

最終更新: 2026-04-19

この台帳は、`engine/` と `vtsimnx/` の横断リファクタリングを実施順で管理するためのドキュメント。
「どこから着手すると効果が大きいか」を明確にし、AI実装時の受け入れ条件を揃える。

## 0. 方針

- 互換優先: 既存 API / 既存 `run_calc` の I/O 契約は維持する。
- 段階移行: 一度に大改修せず、小さく分割して計測・検証する。
- 可観測性先行: まず計測項目を揃えてから構造変更する。
- 二層運用前提: 将来の `remote` / `light` 分離を阻害しない分割を選ぶ。

## 1. High Priority（先に着手）

### H1. `engine/app/main.py` の責務分離

- 対象: `engine/app/main.py`
- 課題: ルーティング、実行パイプライン、artifact処理、エラー整形が同居。
- 方針:
  - `run` パイプラインを `services/simulation_service.py` へ切り出す。
  - artifact API を `routes/artifacts.py` へ分離する。
- 受け入れ条件:
  - `/run` `/artifacts/*` のレスポンス互換が維持される。
  - 既存テストが全通過する。

### H2. `engine/solver/simulation_runner.cpp` の連成ループ分割

- 対象: `engine/solver/simulation_runner.cpp`
- 課題: 連成制御・ログ・反復判定・状態復元が密結合で変更リスクが高い。
- 方針:
  - `performCoupledStepCalculation` と収束判定を独立関数へ分離。
  - ログ組み立てを計算本体から分離する。
- 受け入れ条件:
  - 既存ケースで `solver.log` の収束結果が一致（文言差は許容範囲定義）。
  - 性能退行がない（`simulation_total` が +3% 以内）。

### H3. `CalcRunResult` のストア抽象化（remote/light の境界）

- 対象: `vtsimnx/run_calc/run_calc.py`, `vtsimnx/artifacts/get_artifact_file.py`
- 課題: HTTP取得、schemaキャッシュ、DataFrame復元、エラー処理が一体化。
- 方針:
  - `ArtifactStore` 抽象（`fetch_bytes`, `fetch_schema`, `list_keys`）を導入。
  - `HttpArtifactStore` と `LocalArtifactStore` を分離実装する。
- 受け入れ条件:
  - 既存 `run_calc` の呼び出しコード変更不要。
  - remote モードで挙動互換、light モードを将来追加可能。

### H4. 熱ソルバのキャッシュ状態をコンテキスト化

- 対象: `engine/solver/core/thermal/*`
- 課題: グローバルキャッシュ（LU/トポロジ/統計）が将来の並列化と衝突しやすい。
- 方針:
  - `DirectTSolverContext` を導入し、状態を構造体へ寄せる。
  - テストから明示的に初期化/破棄できるようにする。
- 受け入れ条件:
  - `test_thermal_direct_cache` 系が通過。
  - 既存性能が維持される。

## 2. Medium Priority（次段）

### M1. `engine/app/builder/surfaces.py` の分割

- 課題: RC/応答係数/日射/放射が1ファイルに集中。
- 方針: `surface_rc.py`, `surface_response.py`, `surface_radiation.py` に分割。
- 受け入れ条件: 既存 builder 入力に対して出力差分ゼロ（順序差のみ許容）。

### M2. `solver_runner.py` の責務分離

- 課題: 入力キャッシュ、subprocess起動、manifest処理が単一モジュールに集中。
- 方針: `SolverInvocation` / `ArtifactWriter` / `InputCache` を分割。
- 受け入れ条件: `run_solver` API互換、`test_solver_runner_*` 通過。

### M3. builder オプションの dataclass 化

- 対象: `engine/app/builder/builder.py`, `engine/app/main.py`
- 課題: 同一オプション群の引数が重複し、追加時の修正漏れが起きやすい。
- 方針: `BuildOptions` dataclass を導入し内部経路を統一。
- 受け入れ条件: `/run` の既存リクエスト仕様は据え置き。

## 3. Low Priority（余力時）

### L1. `VTSIMNX_*` 環境変数の集約

- 対象: `engine/app/*`, `engine/solver/app/*`
- 方針: 起動時に設定を一元読み込みし、モジュール内の直接参照を減らす。

### L2. テストのモック契約の明確化

- 対象: `engine/tests_py/test_main.py` など
- 方針: `TypeError` フォールバック依存を減らし、明示インターフェースで差し替える。

## 4. 二層運用への接続タスク（別トラック）

- `backend=remote|light` の切替導線を `run_calc` に導入（デフォルトは remote）。
- `pressureCalc=true` の light 実行は軽量解法（未収束時フォールバック）を設計。
- 受け入れ条件:
  - 既存 remote は完全互換。
  - light は限定機能でも失敗モードが明確。

## 5. 実施順（推奨）

1. H1（`main.py` 分離）
2. H3（`ArtifactStore` 抽象化）
3. H2（連成ループ分割）
4. H4（熱ソルバ文脈化）
5. M1/M2（builder・runner 分割）

## 6. 完了定義（共通）

- 既存回帰テスト通過（Python/C++）。
- `manifest.json` / `schema.json` / `solver.log` の契約互換。
- 代表ケースで `simulation_total` 性能退行が +3% 以内。


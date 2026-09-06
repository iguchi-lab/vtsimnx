# validation strategy

このドキュメントは、`vtsimnx` の検証方針を公開するためのページです。
目的は、実装検証（verification：定めた式を正しく解くか）と妥当性確認（validation：目的とする実現象を十分に再現するか）の根拠と限界を明確にすることである。回帰一致、数値収束、実測一致はそれぞれ異なる証拠であり、相互に代替できない。

## 検証ピラミッド

以下は検証の階層である。下位の項目ほど統合的な比較を扱うが、テスト数や実測検証の完了を示すものではない。

1. **単体テスト（unit）**
   - schema ヘルパ、serialization、artifact クライアント
   - 補助関数（solar/nocturnal/ground/schedule など）
2. **I/O・契約テスト**
   - `/run` リクエスト/レスポンス契約
   - artifact マニフェスト構造
3. **小規模基準ケース**
   - 単純な換気・熱回路での解析可能ケース
   - 数値安定性・収束挙動の確認
   - 公開 CI: `physics-regression`（Python E2E）および `cpp-solver`（C++ known solutions）
4. **数値回帰（regression）**
   - SimHeat / 既存ケースとの比較
   - RC法・応答係数法の比較観点
5. **実測比較ケース（将来拡張）**
   - 実測データとの比較検証
   - 公開可能な条件で再現可能な形に整理

## 小規模基準ケース（CI 対応）

| カテゴリ | CI ジョブ | テスト |
|----------|-----------|--------|
| 単室熱収支（RC golden + エネルギー残差 + 収束） | `physics-regression` | `tests_py/physics/test_thermal_balance.py` |
| RC vs 応答係数（容量なしの等価表面間係数） | `physics-regression` | 同上 |
| 複数室＋換気連成（体積流量収支） | `physics-regression` | `tests_py/physics/test_multiroom_vent.py` |
| 日射・夜間放射 | `physics-regression` | `tests_py/physics/test_solar_nocturnal.py` |
| 湿気応答・非負性・枝流束検査（Phase1） | `physics-regression` | `tests_py/physics/test_humidity_balance.py` |
| 濃度の非負性・枝流束検査 | `physics-regression` | `tests_py/physics/test_concentration_balance.py` |
| 空調 ON/OFF 境界 | `physics-regression` | `tests_py/physics/test_hvac_onoff.py` |
| 収束しにくい圧力網 | `physics-regression` | `tests_py/physics/test_stiff_pressure.py` |
| ゼロ容量・極端抵抗 | `physics-regression` | `tests_py/physics/test_extreme_params.py` |
| 時間刻み変更時の収束性 | `physics-regression` | `tests_py/physics/test_timestep_convergence.py` |
| 換気・熱・湿度の解析解（関数単位） | `cpp-solver` | `vtsimnx_solver_cpp_test_core_physics_known_solutions` |

ローカル実行例:

```bash
cd engine
python -m pytest tests_py -m physics
```

`engine-python` ジョブは builder / API など非物理・非性能テストを `-m "not physics and not perf"` で実行します。

### 物理テストで明示検証する不変条件

golden 一致だけでは誤った結果を固定しうるため、テストごとに次の指標を選択して検証する。全ケースが全指標を検証するわけではない（許容値の根拠は `tests_py/physics/tolerances.py`）。

- **エネルギー収支残差**: `thermal_heat_rate_*` から節点残差を再構成
- **体積流量収支と輸送流束**: `vent_flow_rate` / `humidity_flux` / `concentration_flux`。湿気・濃度の蓄積、生成、除去を含む完全な収支検証とは区別する
- **非負であるべき量**: 絶対湿度・濃度・日射ゲインなど
- **反復回数と収束判定**: `solver.log` の `収束` / `maxBalance` / `総連成反復回数`
- **NaN・Inf の不在**: 全成果物系列 + ログ文言

### 既存テストの範囲を読む際の注意

- `test_rc_vs_response_numeric_regression` の比較壁は容量なしで、応答係数も履歴項を持たない。多層壁の非定常応答や自動生成CTFの精度をこのテストだけで保証できない。
- `test_concentration_mass_balance_nonnegative` は有限性・非負性と、枝流束から作る全節点収支の総和を検査する。各枝を両端で相殺した総和は構成上0になり得るため、蓄積・発生・沈着まで含む独立の物質収支検証にはならない。
- 濃度更新は各室の流入濃度を旧時刻値に固定した局所解析積分である。多室連立系全体の厳密解ではなく、時間刻み感度の確認が必要である。
- 熱流出力は枝種別で符号と意味が異なる。とくに移流の温度差由来の出力を伝導と同じ反対称流束として加算しない。

## 性能履歴（CI: `perf-history`）

KLU / DirectT キャッシュ等の改善を継続監視するため、代表ケースで次を記録します。

- builder 時間 / solver 時間（wall + `timings.simulation_total`）
- 最大メモリ（子プロセス `ru_maxrss`）
- LU 再構築回数（`DirectT cache stats` の `luFactorize` / `topoRebuild`）
- 空調再計算回数（ログ `再計算を実行します`）
- artifact サイズ

厳密な合否はランナー揺らぎで不安定なため、当初は **履歴 JSON の artifact 保存** と **baseline 比で大幅退行時の WARNING** のみとします（ジョブは計測成功で緑）。

```bash
cd engine
python tests_py/perf/run_bench.py --output /tmp/perf_report.json
python -m pytest tests_py -m perf
```

baseline 更新: `tests_py/perf/baselines/representative.json` を意図的な性能変化後に再生成してコミットします。

## 現在の回帰検証対象

- 入力/出力の取り回し（JSON正規化、artifact 取得、主要系列の読み出し）
- 補助計算 API の既存条件における再現性。出典・適用域や外部依存データまで含む一致を一律保証するものではない
- 過去ケースに対する回帰チェックの実行基盤（テストスイート）
- 空調制御の外側ループ回帰（ON/OFF・能力補正・DUCT_CENTRALの処理熱量連動風量補正による再計算要求）
- 湿気計算は Phase1（線形RC）を正式サポート対象とし、移流 + 生成 + `moisture_conductance` + `moisture_capacity` の範囲で回帰保証
- 上記小規模基準ケースの各assertionを公開 CI で実行する構成（個々の実行成功は対象commitのCI結果で確認）
- 代表ケースの性能メトリクスを公開 CI で継続記録

## 現在の未保証・注意点

- あらゆる建物条件での物理妥当性を包括的に保証するものではありません。
- 実測比較の網羅性は現時点で限定的です。
- 空調制御・湿気連成などはケース依存性が高く、利用前の個別検証が必要です。
- 湿気の非線形HAM（吸着等温線ヒステリシス、物性の温湿度依存、液水移動など）は将来拡張対象です。
- 性能の厳密 SLA（絶対 ms 上限）は未設定です。

- 前処理の風下風圧符号、温湿度由来夜間放射の符号、PMV低代謝域の発汗項、ガラス日射の未吸収分など、実装上の要検討事項は [レビュー記録](documentation_review_2026-09.md) を参照する。

## 判定指標（運用目安）

公開版での判定は、次の観点を段階的に整備します。

- **収支残差**: energy balance / mass balance residual（節点流量和など成果物から検証可能な代理指標を含む）
- **比較誤差**: 既存手法との差（時系列・統計指標）
- **再現性**: 同条件再計算での一致性
- **収束**: ログ上の収束フラグと反復回数
- **性能**: baseline 比の大幅退行警告（当初は非 fatal）

しきい値はケース種別ごとに定義し、検証ケース追加時に更新します（`tests_py/physics/tolerances.py`）。

## ドキュメントとの対応

- 利用者向け入力導線: `builder_input_quickstart.md`
- ノード/ブランチ仕様: `node_branch_schema.md`
- engine 実装仕様: `../engine/docs/README.md`
- C++テストの具体例: `../engine/docs/cpp_test_catalog.md`
- Python 物理スイート: `../engine/tests_py/README.md`
- リリース方針: `release_policy.md`

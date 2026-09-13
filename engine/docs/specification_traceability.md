# 仕様・実装・テスト対応表

この文書は、変更時に読む仕様、実装、既存テストを結び付けるための索引です。対象は入力時系列、連成制御、湿度・潜熱、応答係数の安定化です。全機能の検証完了を示す一覧ではありません。

## 文書の読み分け

- **契約・計算仕様**: 入出力、単位、条件、期待する動作。変更する場合は互換性と計算結果への影響を評価します。
- **実装説明**: 関数構成、処理順、キャッシュなど、現在のコードを理解するための説明。関数名そのものは公開 API の保証ではありません。
- **現行の制約・近似**: 対象外の条件、近似、安定化によって失われる性質。意図した物理モデルと区別します。
- **将来案**: 未実装の提案。現在の対応機能や保証として扱いません。

文書とコードが食い違う場合、コードの挙動は「現状の観測結果」として記録し、文書またはコードを自動的に正しいと決めないでください。計算の根拠、公開契約、既存テストを確認し、意図した仕様を決めて修正します。

## 変更時の対応表

パスは各リンクから参照できます。テスト名は検索用です。「検証範囲」は既存テストを読んで確認した観点であり、この文書の更新時にテストを実行したという意味ではありません。

| 対象・期待する動作 | 仕様 | 実装 | 既存テスト・検証範囲 |
|---|---|---|---|
| 時系列の長さ 1 を展開し、不一致を拒否する | [builder入力 §5](builder_json.md) | [validate.py](../app/builder/validate.py): `_normalize_node_series_fields` ほか | [test_validate.py](../tests_py/builder/test_validate.py): `test_node_timeseries_broadcasts_length_one`, `test_node_timeseries_rejects_mismatched_length`, `test_vol_timeseries_rejects_mismatched_length` |
| 内側連成の不要判定・最小反復・未収束終了 | [計算ループ §5.1](simulation_loops.md) | [simulation_coupling_control.cpp](../solver/simulation_coupling_control.cpp): `evaluateInnerCoupling` | [test_simulation_control_flow.cpp](../solver/tests_cpp/test_simulation_control_flow.cpp): `testInnerCouplingBreakNoNeed`, `testInnerCouplingConvergedOnSecond`, `testInnerCouplingMaxIterations`, `testInnerCouplingPressureFirstFail`。判定関数の分岐を検証 |
| 外側上限と反復上限のフォールバック | [計算ループ §4–5](simulation_loops.md) | [simulation_runner_helpers.cpp](../solver/simulation_runner_helpers.cpp), [simulation_coupling_control.cpp](../solver/simulation_coupling_control.cpp) | [test_simulation_control_flow.cpp](../solver/tests_cpp/test_simulation_control_flow.cpp): `testEffectiveIterationFallback`, `testOuterMaxIterationsThrow`, `testOuterAcceptDoesNotThrow` |
| 潜熱は湿度・温度・内側湿度連成が有効なときだけ適用する | [計算ループ §5.2](simulation_loops.md) | [simulation_inner_coupling.h](../solver/simulation_inner_coupling.h): `latentCouplingActive`; [sim_constants_parser.cpp](../solver/parser/sim_constants_parser.cpp) | [test_simulation_control_flow.cpp](../solver/tests_cpp/test_simulation_control_flow.cpp): `testLatentCouplingActiveRequiresHumidity`。同ファイルに非連成や温度 OFF と潜熱モードの併用を拒否するパーサ検証あり |
| 非連成湿度の未知ノードは時間積分基準をステップ開始時に固定し、固定境界は更新を保持する | [計算ループ §5.2](simulation_loops.md) | [simulation_inner_coupling.cpp](../solver/simulation_inner_coupling.cpp): `runDecoupledHumidityStep` | 実装を確認済み。この動作を直接検証する専用テストは今回の確認範囲では未確認。変更時は外側反復回数を変えても時間を二重に進めないケースを確認する |
| 応答係数生成で `sum_c > 0.9999` のとき履歴を除いて定常 U に置換する | [物理・数学メモ §4.3](physics_math_notes.md) | [surface_response.py](../app/builder/surface_response.py) | [test_surfaces_response_conduction.py](../tests_py/builder/test_surfaces_response_conduction.py): `test_auto_response_coefficients_falls_back_to_steady_state_when_sum_c_near_1`。遅い系のフォールバックを検証。閾値の全境界や過渡応答同等性の保証ではない |

物理的な妥当性の保証範囲は [validation_strategy.md](../../docs/validation_strategy.md)、主要 C++ テストの観点は [cpp_test_catalog.md](cpp_test_catalog.md) を参照してください。関数の分岐テストと、シミュレーション全体の収支・精度の検証は区別します。

## 文書を更新する手順

1. 変更対象の入力条件・期待結果・異常時の動作を、上表の仕様正本に記載する。例や利用ガイドには要点と正本へのリンクを置く。
2. 実装説明と制約を見直す。数値近似やフォールバックでは、目的に加えて失われる性質と適用条件を記載する。
3. 対応するテストの実際の assertion を確認し、変更した仕様を検証できるか判断する。未検証の条件はその旨を記録し、合格済みと扱わない。
4. 影響する既存テストを実行し、必要な場合に境界値・収支・回帰の検証を追加する。文書だけの修正ではコード照合、リンク確認、差分確認を行い、未実行のテストを明記する。
5. コード変更と関連文書を同じ変更単位で管理する。公開互換性の扱いは [public_api.md](../../docs/public_api.md) と [release_policy.md](../../docs/release_policy.md) に従う。

## 今回の確認記録

- 確認日: 2026-09-13。
- 対象: この作業コピーの上表の実装・テスト。push 前に `bbf924b` の main と再照合し、非連成湿度の未知ノード復元と固定境界保持を反映。別バージョンへの適用は再照合が必要。
- 方法: ソースとテストの読解。文書修正に伴う数値テストの実行は行っていない。
- 未確認: 全機能の網羅性、Word 版文書との同期、全テストの合格状況。

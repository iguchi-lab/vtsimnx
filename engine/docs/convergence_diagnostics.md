# 収束診断と失敗入力の再実行

## 診断と誤収束防止（bc9c15b）

この節は最初の診断・熱解法対応の範囲を示す。Issue #24 の追加修正は末尾を参照。

非物理的な温度・絶対湿度の候補は、ノード名と値を付けてエラーにする。元の値へ戻して成功と扱わない。連成中の温度・湿度は緩和前の候補差も収束判定に用いる。設定した物理範囲、許容誤差、運転制御の閾値、潜熱モデルの式は変更していない。

この変更は誤収束防止・熱方程式の解法修正・診断基盤であり、年間計算の除湿振動を解消したという意味ではない。最終状態での保存則の一括評価、状態更新の一元化、加速法の導入は別途検証する。

## 失敗時の保存

solver がエラーを返す、クラッシュする、出力 JSON が読めない場合、`engine/work/runs/<run_id>/replay/` に以下を保存する。

- `input.json.gz`: solver に渡したバイト列そのまま（builder 展開後）。
- `replay.json`: 入力と solver 実行ファイルの SHA-256、run ID、失敗理由、時刻。

通常成功時には作らない。API の後処理でも残り、公開ダウンロードのファイル一覧には追加しない。保存失敗はログに記録し、元の solver エラーを隠さない。`VTSIMNX_KEEP_RUN_FILES=1` の場合は通常の input/output も API 後処理で保持する。

replay は調査用のローカル保存で、既存の artifacts 容量・期限管理の対象外。調査終了後の保管・削除は運用側で管理する。入力だけでは実行ファイルを復元できないため、検証するバイナリも別途保存する。入力保持は既に削除された過去の入力を復元しない。

## 再実行

元の計算ディレクトリを使わず、新しい出力ディレクトリを指定する。

```bash
cd engine
python3 tools/replay_failure.py work/runs/<run_id>/replay \
  --solver /path/to/original/vtsimnx_solver --out /path/to/new/replay-output
```

同じ solver バイナリかハッシュで確認する。修正版との比較時だけ `--allow-different-solver` を付ける。入力のハッシュ不一致は常に停止する。既存出力ディレクトリへの上書きは拒否する。再現した計算自体がエラーならコマンドも非ゼロで終了する。

これは入力全体の再実行であり、途中状態からの再開ではない。年間計算を途中から再現するには、壁体・材料・空調状態を含むチェックポイントが別途必要。現時点では年月日の入力切り出しで代用しない。

## 湿度境界のログ

`latent_boundary` には更新元 `stage=capacity/airflow`、実際の潜熱モデル分岐 `regime`、吸込・吹出温度、吸込湿度、変更前後の吹出湿度、目標RH、風量、顕熱・潜熱、凝縮量、再計算要求を記録する。verbosity 1 では湿度変更時、2 では各評価時に出す。

潜熱がゼロと非ゼロを往復する際は、同じ時刻で `regime` と `x_in/x_old/x_new` を比較する。例えば `coil_dry_or_inactive` と `coil_wet` の交互変化なら、湿度更新の上書きとは区別してコイルの分岐条件を調べる。診断のために潜熱をゼロへ固定したり、許容誤差を広げたりしない。

## 回帰検証

```bash
cd engine
PYTHONPATH=.. python3 -m pytest tests_py/test_convergence_safety.py tests_py/test_failure_replay.py
PYTHONPATH=.. python3 -m pytest tests_py/physics
```

`VTSIMNX_TEST_SOLVER=/path/to/baseline` を指定すれば同じ安全性テストを修正前バイナリにも適用できる。今回追加した異常温度、異常湿度、極小緩和の7ケースは修正前に成功を返してしまうことを確認した。

## 冷房ON/OFF振動の解法修正

1室の冷房で room 温度が約30.3553°C（OFF解）と12.4746°C（ON解）を往復し、100反復で失敗するケースを保存入力から再現した。基準バイナリ（0fc1d1e時点）でも同じエラーになる。

原因は、熱行列の非ゼロ配置だけで対称性を判定していたこと。空調の設定温度行を組み込んだ行列は、配置が対称でも係数値が非対称になる。LLT/LDLT は片側の三角部分から対称行列として計算するため、別の式の解を返していた。さらに全行の最大RHSで正規化した残差判定では、熱収支[W]の大きさが温度拘束[K]の誤差を覆い隠す。後処理の熱収支にも設定温度との差の検査がなかった。

係数値も対称な場合だけ Cholesky 経路を選び、それ以外は既存の KLU/Eigen SparseLU を使用する。設定温度の拘束は、熱収支のスケールとは別に温度の許容誤差で検査する。キャッシュ解にも同じ検査を適用する。係数の厳密比較で微小な非対称もLUへ回すため、対象ケースによって性能への影響はあり得る。

`test_cooling_setpoint_does_not_cycle` は KLU/Eigen の両経路で成功と初回26°Cを検証する。修正前に strict xfail としたケースを通常の回帰テストへ変更した。このON/OFF振動を年間計算の潜熱振動と同一原因とは扱わない。

解法の前提は [Eigen の疎行列ソルバー説明](https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html) と [SimplicialLDLT の仕様](https://libeigen.gitlab.io/eigen/docs-master/classEigen_1_1SimplicialLDLT.html) を参照。

## 2026-11-12 の検証結果

- 基準ソース: `0fc1d1eafbff1512ad3d8c3254be64106efdee2f`。
- Release の `vtsimnx_solver` ターゲットを `-j2` でビルド成功。
- `PYTHONPATH=.. python3 -m pytest tests_py -m 'not perf' -q -o addopts= --tb=short`: 270 passed、性能テスト1件を除外。未解決を隠すxfailは使用していない。環境のrequests依存バージョン警告が1件ある。
- 変更したPythonファイルのruff検査と `git diff --check` に合格。
- 保存した冷房入力（SHA-256 `dc166ef644acf72035a97512335f215cbf75c5c10549bf9d9b45e20e1dcf7da0`）は基準バイナリで `aircon_max_iterations`、最終バイナリで `ok`。入力バイト列を変更せず比較した。
- C++全テストターゲットと年間ケース・性能ベンチマークは未実施。年間ケースは元の実行入力が必要。
- 検証用コピー: `/home/ubuntu/work/vtsimnx-convergence-20261112`、ブランチ `fix/convergence-diagnostics`。稼働中の `/home/ubuntu/vtsimnx` は変更していない。

## 2026-11-13: Issue #24 の除湿不連続を修正

コイル面積法の暫定出口湿度を `xe = min(xr, max(0, x_sat(te)))` とした。冷却のみで暫定湿度が吸込より増える仮定を除き、露点境界で潜熱量が0から連続に立ち上がるようにする。根拠と式は [空調モデル10.4.5節](../../docs/aircon_model_and_control.md) を参照。

### 再現と検証

- C++実関数で、温度・顕熱・風量・面積の4条件、方式名3種類、湿りエンタルピー有効/無効を組み合わせた24条件の境界を探索。境界の上下、出口飽和湿度との切替点、湿度に対する潜熱の単調性、吹出湿度の範囲、凝縮水量と湿度差の一致を検査する。
- 代表条件（吸込27.5°C、顕熱4725 W、風量0.3 m³/s、既定面積）で、露点境界から `1e-8 kg/kg(DA)` 湿った側の潜熱は、修正前約1267.7 W、修正後約0.0113 W。それぞれの実装の境界を探索して測定した値であり、同じ絶対湿度点での単純比較ではない。
- 1室の湿度フィードバック6条件は、基準バイナリ `bc9c15b` で全件 `aircon_max_iterations`、修正版で全件成功。全館空調 `DUCT_CENTRAL` の湿度を変化させる6時間ケースも、顕熱モード/湿りエンタルピーモードとも成功。
- 保存した失敗入力を `tools/replay_failure.py` でバイト列を変えず再実行し、修正版で `ok` を確認。
- 未指定の `pre_rh` はログ上で `unset` と表記する。内部の未指定値NaNを、計算異常のNaNと誤認させないための表示変更。

検証コマンド:

```bash
cd engine
cmake --build build-solver --target vtsimnx_solver vtsimnx_solver_cpp_test_aircon_controller -j2
ctest --test-dir build-solver -R '^vtsimnx_solver_cpp_test_aircon_controller$' --output-on-failure
PYTHONPATH=.. python3 -m pytest tests_py -m 'not perf' -q -o addopts= --tb=short
```

結果はPython **278 passed**（性能テスト1件除外）、対象C++テスト **1/1 passed**。変更Pythonのruff検査と差分形式検査も成功。C++全ターゲットの再検証はしていない。

### 残る確認範囲

今回の比較では、出口湿度への追加の緩和や許容誤差の拡大は不要だった。既存の吹出湿度再計算閾値 `1e-4 kg/kg(DA)` は変更しておらず、最終状態の保存則をこの閾値より厳密に一括保証したという意味ではない。

小規模再現での不連続・反復上限の解消と、年間ケース全条件の解決確認は区別する。係数の実測妥当性や適用風速域も、この修正の検証対象外。

### 稼働反映と年間比較の追試（2026-11-13）

`4b84e4c9b9148ccb4a4b434de4900837fe8fa57c` の修正は稼働版へ反映済み。ソルバーのSHA-256は `64a637c8dcfa3d9b1fedec4436e0445aff7d679b3c95048a89ff2ace22687858` で、作業用・稼働用バイナリの一致とAPI経由の6時間ケースを確認した。

その後、元のColab比較に使った年間入力を保存アーカイブから取得した。従来の3方式×C値6条件について、入力ハッシュ18件が旧計算と完全一致することを確認し、同じ稼働バイナリで追試している。収束許容値や反復上限は変更していない。

- C値0.00、0.30、0.59では、`coil_aoaf`・`rh95`・`bf`の全方式で年間8,760時間の計算と出力取得を確認した。
- 追加した既存の理想除湿条件（冷房時`pre_rh=60`、`latent_method=none`）も、上記3つのC値で年間出力を取得した。この条件は吸込温度基準の除湿で、室内RHを60%に固定する制御ではない。
- C値4.74は4方式とも64ステップ付近で`thermal_not_converged`となった。Issue #24の露点境界振動が全条件で解消した証拠とは扱わず、保存された失敗入力とログから別途調査する。
- 完了条件にも、AC出口の温度・絶対湿度から求めたRHが100%を超える時刻がある。値は切り詰めず記録し、最終出力の温湿度・収支の整合を残る確認事項とする。

これは追試途中の記録であり、年間24条件の全成功や物理的妥当性の全面確認を意味しない。年間入力・結果・APIキーは公開リポジトリへ含めない。

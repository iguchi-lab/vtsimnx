# C++テストカタログ（具体例つき）

このドキュメントは `engine/solver/tests_cpp` の主要テストについて、  
「何を検証しているか」「どんな具体ケースで検証しているか」「何を合格基準にしているか」をまとめた一覧です。

## 1. 使い方

- 単体実行例（1本）:
  - `cmake --build engine/build-solver --target vtsimnx_solver_cpp_test_acmodel_core -j4`
  - `engine/build/vtsimnx_solver_cpp_test_acmodel_core`
- まとめて実行:
  - `ctest --test-dir engine/build-solver -j1 --output-on-failure`

## 2. テスト一覧（要点）

### 2.1 流体・換気コア

- `vtsimnx_solver_cpp_test_flow_math`
  - 検証内容: 開口/隙間/固定流量/ファンの流量式とヤコビアン整合
  - 具体例:
    - `simple_opening` で `Q(-dp) = -Q(dp)`（奇関数性）
    - `gap (n=1)` で `Q=a*dp` の線形一致
    - `fan` でしきい値近傍を含む複数 `dp` 点で解析ヤコビアンと数値微分一致
  - 合格基準: 各点で許容誤差内（例: `1e-6` など）

- `vtsimnx_solver_cpp_test_fan_jacobian_sweep`
  - 検証内容: ファンヤコビアンのランダムスイープ回帰
  - 具体例:
    - 乱数で `p_max, p1, q_max, q1` を50ケース生成
    - 1ケース12点（しきい値近傍 + 広域）で解析ヤコビアンと数値微分比較
  - 合格基準: すべての点で差が `2e-3` 以内

- `vtsimnx_solver_cpp_test_pressure_constraints`
  - 検証内容: Ceres制約（残差/Jacobian）の符号・微分正当性
  - 具体例:
    - 2ノード1枝（A-B）の流量収支で、A残差は `-Q`、B残差は `+Q`
    - `SoftAnchorConstraint` の勾配が重みと一致
    - 欠損ノード指定時は残差/Jacobianがゼロ
  - 合格基準: 解析式と数値微分が許容誤差内で一致

- `vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates`
  - 検証内容: 並列枝の流量が枝ごとに保持されること
  - 具体例:
    - A->B 並列2枝で係数 `a` を変える（0.01 / 0.02）
    - 差圧100Pa時に2枝の `flow_rate` が別値になることを確認
  - 合格基準: 各枝の `flow_rate` が `calculateUnifiedFlow` 再計算値に一致

### 2.2 パーサ/入力バリデーション

- `vtsimnx_solver_cpp_test_parser_thermal_branches`
  - 検証内容: `response_conduction` 必須項目/係数長整合
  - 具体例:
    - `area` 欠損で例外
    - `resp_a_src` と `resp_b_src` 長さ不一致で例外
    - 正常データで係数配列が期待どおりにパース
  - 合格基準: 異常系は例外、正常系は値一致

- `vtsimnx_solver_cpp_test_parser_branches_validation`
  - 検証内容: branch `key` 必須、重複禁止、`enable` 互換
  - 具体例:
    - `ventilation_branches` の `key` 欠損で例外
    - 重複 `key` で例外
    - `enable` が bool/配列/未指定（デフォルトtrue）を時刻別に確認
  - 合格基準: 期待どおりに throw / 非throw

- `vtsimnx_solver_cpp_test_parser_nodes_mode`
  - 検証内容: `nodes[].mode` の後方互換（数値→文字列）
  - 具体例:
    - `mode: 1` -> `HEATING`
    - `mode: [0,1,2,3]` の時刻参照
    - 混在配列（`"OFF"`, `3`）の変換
    - 不正コード `9` で例外
  - 合格基準: 変換結果文字列が期待値

- `vtsimnx_solver_cpp_test_parser_nodes_required`
  - 検証内容: node の必須キー検証
  - 具体例: `key` 欠損、`type` 欠損をそれぞれ例外確認
  - 合格基準: 必須欠損時に必ず例外

### 2.3 熱計算・ネットワーク連成

- `vtsimnx_solver_cpp_test_thermal_linear_utils`
  - 検証内容: 係数行インデックス変換・対称パターン判定
  - 具体例:
    - `RowIndexMap` で `cols={0,2,5,9,...}` の逆引き
    - 対称/非対称の疎パターン判定
  - 合格基準: 逆引きインデックス一致、真偽一致

- `vtsimnx_solver_cpp_test_thermal_direct_cache`
  - 検証内容: 直接法ソルバのキャッシュ有効化/無効化
  - 具体例:
    - 1回目 `fullBuild + solveFull`
    - 同条件2回目 `rhsOnlyBuild + solveCached`
    - エアコンON/OFF切替や移流流量変更で再び `fullBuild`
  - 合格基準: 統計カウンタ遷移が期待どおり

- `vtsimnx_solver_cpp_test_thermal_advection_sign`
  - 検証内容: 負流量時の移流符号整合
  - 具体例:
    - `A->B` で `flow<0` と `B->A` で `flow>0` の等価ケース比較
    - 収支RMSEが双方ゼロかつ一致
  - 合格基準: RMSE 0 近傍（`1e-8`以内）

- `vtsimnx_solver_cpp_test_network_rebuild`
  - 検証内容: `buildFromData` の再実行で重複蓄積しないこと
  - 具体例:
    - Ventilation/Thermal を2回連続構築して node/edge 数が不変
    - 重複 source-target の移流枝で flow_rate が枝ごとに保持
  - 合格基準: ノード数・枝数・flow_rate が期待どおり

- `vtsimnx_solver_cpp_test_transport_humidity_concentration`
  - 検証内容: 湿度/濃度1ステップ更新
  - 具体例:
    - 湿度: implicit更新式  
      `x_{n+1} = (x_n + a*x_src)/(1+a)` と一致確認
    - 濃度: 崩壊項のみ  
      `c(t+dt) = c(t) * exp(-beta*dt)` と一致確認
    - 濃度: 沈着なし/あり（`beta=0` / `beta>0`）で既知解一致
    - 濃度: `eta` 付き流入（`q*(1-eta)`）と符号付き流量（`q<0` 逆流）を既知解で検証
  - 合格基準: 数式ベース期待値と高精度一致

### 2.4 空調モデル・制御・アプリ

- `vtsimnx_solver_cpp_test_refrigerant_calculator`
  - 検証内容: 冷媒物性・理論効率の単調性/回帰
  - 具体例:
    - 飽和圧の温度単調増加
    - 理論暖房効率が温度リフト増加で低下
    - 代表固定値（`p_sat(-20C)` など）の数値回帰
  - 合格基準: 単調性成立、固定値が許容差内

- `vtsimnx_solver_cpp_test_acmodel_core`
  - 検証内容: `CRIEPI/RAC/DUCT_CENTRAL` の回帰
  - 具体例:
    - CRIEPI: JIS点で `power` が仕様値近傍
    - RAC: 参照データ（抜粋/全件）との統計誤差監視
    - DUCT_CENTRAL: 暖房/冷房代表点で  
      送風機・圧縮機・合計を個別回帰、`V_outer` 分離、デフロスト境界、`V_vent=0`同値
  - 合格基準: 各回帰値・統計閾値・境界条件を満たす

- `vtsimnx_solver_cpp_test_aircon_controller`
  - 検証内容: 制御順序、ON/OFFゲート、潜熱手法、能力超過補正
  - 具体例:
    - OFF機器は `power=0`、ONのみ `estimateCOP` 呼び出し
    - `AUTO` モードで `indoorTemp` と `airconTemp` の関係から `cooling/heating` が分岐
    - `coil_aoaf` で `Ao` 変更時に `Q_L` 変化
    - 過負荷時に `current_pre_temp` が補正されること
    - 複数機器が同じ `set_node` を持つ場合、潜熱フィードバック注入を抑止
    - `in_node` 不正時に例外で停止せず `power=0` で継続
  - 合格基準: 呼び出し回数・値・補正方向が期待どおり

- `vtsimnx_solver_cpp_test_app_smoke`
  - 検証内容: `runVtsimnxSolverApp` の最小E2E
  - 具体例:
    - 最小入力で `status=ok`、`artifact_dir` と `schema.json` 生成
    - 2ステップケースで `applyPreset` ログが毎ステップ出ることを確認
  - 合格基準: 返り値0、出力JSON/成果物/ログ条件を満たす

## 3. DUCT_CENTRAL詳細への導線

- DUCT_CENTRAL 実装/回帰の詳細は `docs/duct_central_model_validation.md` を参照。

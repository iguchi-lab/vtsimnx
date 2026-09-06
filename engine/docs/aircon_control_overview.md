### エアコン制御の概要

このドキュメントは、solver 側のエアコン制御が

- どのノード/ブランチを使うか
- 1 タイムステップの中でいつ動くか
- ON/OFF 判定をどうしているか
- 処理熱量・能力上限をどう扱うか

をまとめたものです。

主な実装箇所:

- `solver/aircon/aircon_controller.cpp`
- `solver/simulation_runner.cpp`
- `solver/core/thermal/thermal_direct_build.cpp`
- `solver/core/thermal/thermal_direct_rhs.cpp`

---

### 特徴: エアコン未設置空間の温度制御（遠隔 set）

VTSimNX の空調制御の大きな特徴は、**エアコンの吸込・吹出がある空間と、温度制御の対象空間を分離できる**ことです。

- `set`（`set_node`）: 設定温度で拘束・制御する室（例: LDK）
- `in` / `out`（`in_node` / 吹出先）: 実機の還気・吹出がある空間（例: 階間）

エアコン本体は階間などに置き、階間→LDK などの**別換気・伝導**で空気や熱が伝われば、LDK 側の温度を制御できます。  
床下空調・階間空調・ダクト経由の間接制御など、実建物でよくある構成をそのままモデル化するための設計です。

```mermaid
flowchart LR
    subgraph remote["遠隔 set（特徴的な構成）"]
        IN2["in/out = 階間"] --> AC2["aircon"]
        AC2 --> IN2
        IN2 -->|"別換気・伝導"| SET2["set = LDK<br/>温度制御対象"]
    end
    subgraph local["通常の再循環"]
        R["in/out/set = 同一室"] --> AC1["aircon"]
        AC1 --> R
    end
```

入力例（builder）: `set: "LDK"`, `in`/`out: "階間"`。省略時の `in`/`out` は `set` と同じ（再循環）です。  
負荷推定（`required_heat_w`）が遠隔 set でも破綻しないことは §3 を参照してください。

---

### 1. エアコンノードの役割

builder で `aircon` を与えると、solver 形式では `type="aircon"` のノードと、送風を表す換気ブランチが追加されます。

主な項目:

- `key`: エアコンノード名
- `in_node`: 還気側（実機の吸込空間）
- `set_node`: **設定温度をかける対象室**（吸込空間と異なってよい）
- `outside_node`: 外気条件参照先
- `mode`: `OFF` / `HEATING` / `COOLING` / `AUTO` の時系列
- `pre_temp`: 設定温度の時系列
- `ac_spec`: 能力・消費電力・風量などの仕様

`pre_temp` は solver 側では `vector<double>` として保持し、各 timestep で要求設定へ展開されます（実効設定は能力制限で動き得ます。§9.1）。

```mermaid
flowchart LR
    IN["in_node<br/>還気"] --> AC["aircon ノード"]
    AC --> BLOW["吹出先<br/>（多くは in と同空間）"]
    SET["set_node<br/>室温拘束・制御対象"]
    BLOW -.->|"直結または別換気"| SET
    OUT["outside_node"] -.-> AC
    PRE["pre_temp スケジュール"] --> REQ["requested"]
    REQ --> EFF["effective<br/>current_pre_temp"]
    EFF --> SET
```

---

### 2. タイムステップ内での位置づけ

エアコン制御は、圧力-熱の連成計算が一度落ち着いた後に実行されます。
吹出湿度 `supplyX` が変化した場合も外側ループで再計算し、**同一タイムステップの湿度連成へ反映**します。

```mermaid
flowchart TD
    updateProps[時変プロパティ更新]
    coupledSolve[圧力と熱を連成計算]
    updateHumidity[湿度 x 更新]
    airconControl[エアコン制御とsupplyX適用]
    maybeRecompute{再計算が必要?<br/>風量/ON-OFF/能力/吹出湿度}
    updateConcentration[濃度 c 更新]
    writeResult[結果出力]

    updateProps --> coupledSolve
    coupledSolve --> updateHumidity
    updateHumidity --> airconControl
    airconControl --> maybeRecompute
    maybeRecompute -->|yes| coupledSolve
    maybeRecompute -->|no| updateConcentration
    updateConcentration --> writeResult
```

入口は `solver/simulation_aircon_iteration.cpp` の `runAirconIteration()` です。ループ全体の位置づけは [`simulation_loops.md`](simulation_loops.md) を参照してください。

1. `controlAllAircons()` で ON/OFF を決める
2. ON が安定していれば `checkAndAdjustDuctCentralAirflow()` で DUCT_CENTRAL の風量補正を確認する
3. 全台の ON/OFF が安定していれば `checkAndAdjustCapacity()` で能力超過を確認する
4. 各段階は `AirconStateProposal` を積み上げ、`AirconRecomputeReason` を OR 集約する
5. 優先順位 ON/OFF > Flow > Capacity > SupplyHumidity で再計算 or Accept を決める

メトリクスには従来の種別カウンタに加え、次を記録します。

| キー | 意味 |
|---|---|
| `aircon_recompute_reasons_mask` | 観測した `AirconRecomputeReason` の OR |
| `outer_iterations` | 全タイムステップの外側反復の合計 |
| `outer_iterations_max` | 1ステップあたりの最大外側反復 |
| `outer_iterations_mean` | 平均外側反復（`outer_iterations / solve_timesteps`） |
| `outer_iterations_ge3` / `_share` | 外側反復が 3 以上だったステップ数とその割合 |
| `aircon_capacity_recalc_share` | 空調再計算のうち能力制限起因の割合 |

能力固定モード導入の判断目安: `outer_iterations_mean` が通常 1〜2 なら大改修の効果は限定的。`outer_iterations_ge3_share` が高く、かつ `aircon_capacity_recalc_share` が大きいときに効果が大きい。

---

### 3. ON/OFF 判定

`controlAllAircons()` は次の優先順で ON/OFF を決めます。

1. **ON かつ** 熱ソルバが `required_heat_w`（符号付き必要負荷）を算出済み → **負荷の符号**で判定
2. それ以外（OFF 中、または負荷未評価）→ `set_node` 室温と **要求設定** `current_requested_pre_temp` の温度バンド

符号付き必要負荷（暖房正・冷房負）は、fixed-row 解の後に **設定温度を維持するために空調が担うべき負荷**です。

還気・吹出が `set_node` に直結しているとき（通常の再循環）:

```text
qOther = set_node 熱収支のうち空調ブランチ以外
required_heat_w = -qOther
```

`set_node` と吸込・吹出が別室のとき（**遠隔 set**。冒頭の特徴節を参照。例: set=LDK、in/out=階間）は、
dual-row 後の `set_node` 熱収支が ≈0 になり上記式は常に `Qreq≈0` になるため、
還気→吹出のコイル処理熱量 `ρ·cp·|V|·(Tsupply−Treturn)`（湿り時はエンタルピー差）で代替します。

直結ケースではコイル熱を ON/OFF 正本に使いません（吹出温度が病的なときに符号が狂い得るため）。
固定後の室温は常に設定付近なので、温度比較だけでは「暖房不要なのに ON 維持」を検出できません。

```mermaid
flowchart TD
    M{"mode"} -->|OFF| Z["強制 OFF"]
    M -->|ON + Qreq あり| Q{"符号付き必要負荷"}
    Q -->|HEATING かつ Qreq > +tol| ON1["ON 維持"]
    Q -->|HEATING かつ Qreq ≤ +tol| OFF1["OFF"]
    Q -->|COOLING かつ Qreq < −tol| ON2["ON 維持"]
    Q -->|COOLING かつ Qreq ≥ −tol| OFF2["OFF"]
    M -->|OFF 中 / Qreq なし| T["室温 vs 要求設定（deadband）"]
```

- 暖房: `Qreq > tol` なら ON。`Qreq ≤ tol`（冷房需要・ほぼゼロ含む）なら OFF
- 冷房: `Qreq < -tol` なら ON。それ以外は OFF
- OFF 中の再起動は従来どおり温度バンド（帯内は現状維持）

注意:

- `set_node.calc_t` は ON/OFF に応じて切り替えていません
- 実際の固定温度化は熱ソルバ側の fixed-row ロジックで行います（値は実効設定 `current_pre_temp`）
- 同一 `set_node` を複数空調が制御する入力は `initializeModels()` で拒否します
- 能力 bracket の最終検証でも上限を満たせず拡張できない場合は `CapacityConstraintUnresolved` で例外終了します（超過のまま Accept しない）
- 温度バンド幅は `max(空調温度許容誤差, 0.5K)`。入力 tol が `1e-6` など極小でも、Qreq≈0 で OFF した直後のわずかな温度浮きで再 ON しない
- **能力制限中**（`CapacityLimited`、または実効設定が要求から離れている）は `required_heat_w` で OFF しません。実効設定を下げた拘束では Qreq が符号反転し、OFF↔ON 振動するためです。このときは要求設定との温度バンドで判定します。
---

### 4. 熱ソルバとの接続

エアコンが ON のとき、`set_node` は熱ソルバ内で固定温度行として扱われます。

固定温度に使う値:

- `graph[v_ac].current_pre_temp`（実効設定）

主な参照箇所:

- `solver/core/thermal/thermal_direct_build.cpp`
- `solver/core/thermal/thermal_direct_rhs.cpp`

このため、エアコン制御が `current_pre_temp` を更新すると、次の再計算ではその値が新しい境界条件として使われます。

---

### 5. 処理熱量の定義（顕熱・潜熱）

処理熱量は、内部的に次の 2 つを区別して扱います。

- 顕熱 `sensibleHeatCapacity` [W]
- 潜熱 `latentHeatCapacity` [W]

`AirconController::calculateHeatCapacity()` は顕熱の基礎量を計算します。

概念的には次です。

- `sensible = rho_air * cp_air * |flowRate| * deltaT`（有効な向きのみ）

ここで:

- `flowRate`: `in_node -> airconNode` の流量
- **暖房時**: `deltaT = outletTemp - inletTemp`。**出口温度 ≤ 入口温度のときは 0**（加熱していない）
- **冷房時**: `deltaT = inletTemp - outletTemp`。**入口温度 ≤ 出口温度のときは 0**（除熱していない）

顕熱・潜熱ともに「処理熱量の大きさ [W]」として正値で扱います。  
acmodel 入力では `Q_S=顕熱`, `Q_L=潜熱`, `Q=Q_S+Q_L` を渡します。

`simulation.coupling.moist_enthalpy_enabled=true` のとき、全熱の正本は

\[
Q = \dot m\,|h_\mathrm{in}-h_\mathrm{out}|
\]

（\(h=\) `archenv::total_enthalpy_from_x`、モード向きのみ正）です。  
吹出湿度 `supplyX` は従来どおり `latent_method` で決め、その後に全熱をエンタルピーから再計算し、顕熱/潜熱へ分解して出力・acmodel へ渡します。

---

### 6. 潜熱計算（latent_method）

冷房時の吹出絶対湿度 `supplyX` と潜熱は、`ac_spec.latent_method` で方式を切り替えます。

#### 6.0 理想相対湿度制御（`pre_rh`）

`aircon.pre_rh`（または空調ノードの `pre_rh`）に設定相対湿度 [%] を与えると、**冷房運転中**かつ室絶対湿度が目標を超える場合に、コイルモデルより優先して理想除湿します。

- 目標絶対湿度: `x_sp = absolute_humidity(T_in, pre_rh)`
- 条件: `x_in > x_sp` のとき `supplyX = x_sp`（能力上限・機種特性は見ない）
- `pre_rh` 未指定、または目標以下のときは従来どおり `latent_method` を使用
- 暖房 / OFF では無効（既存のパススルー／除外ロジックのまま）

**方式一覧（冷房時のみ有効）**

- `rh95`（**デフォルト**）  
  吹出温度 `Tout` が決まったら、吹出空気を **RH=95%** とみなして `supplyX` を決定します。
- `bf`  
  バイパスファクタ法で `supplyX` を計算します（`bf`/`BF`/`bypass_factor`、既定 `0.2`）。  
  BF 法で求めた吹出 RH が 100% を超えた場合は、警告を出して **RH95 法へフォールバック** します。
- `coil_aoaf`（別名: `"aoaf"`, `"literature"`）  
  既往文献に基づく **コイル前面風速・有効表面積を用いた方式**です。
  - 入力: 顕熱処理量 `Hs`、吸込温度/絶対湿度、吹出温度、風量 `V` など
  - パラメータ:
    - `Af` または `coil_face_area` : 実質コイル前面面積 [m²]（既定 `0.133`）
    - `Ao` または `coil_surface_area` : コイル有効表面積 [m²]（既定 `4.84`）
  - 手順（概略）:
    - 吸込条件と顕熱負荷から吹出条件（仮の `Te, Xe`）を決める
    - 中間状態 `T*, X*` を取り、前面風速 `Vx = V / Af` から潜熱伝達率 `Kx` と対流熱伝達率 `αc` を求める
    - コイル表面温度 `Td` を算出し、その飽和絶対湿度 `Xd` との差から除湿量 `Hr` [W] を評価
    - `Hr` から水蒸気質量流量を逆算し、出口絶対湿度 `supplyX` を決める
- `none`  
  潜熱処理なし（`Q_L=0`、`supplyX = X_in`）。

いずれの方式でも、計算された `supplyX` は aircon ノードの `current_x` に反映され、  
次ステップおよび `humidity_x` 出力の初期値として利用されます。  
あわせて除湿量 \(\dot m(x_\mathrm{in}-x_\mathrm{supply})\) をノードの `aircon_moisture_removal_kg_s` に保持し、湿気収支診断の `airconCondensation` に載せます。  
吹出側の湿気移流は空調ノードの `current_x`（= supplyX）を**固定境界**として使うため、能力計算・湿気状態・熱移流の湿度が一致します。  
そのため空調ノードは常に `calc_x=false` です（湿度ソルバの未知数にしない）。`calc_x=true` だと supplyX 適用後に湿度解が上書きし、外側ループが `SupplyHumidityChanged` で振動します。  
外側再計算の判定床は `1e-4 kg/kg(DA)`（連成湿度 tol との大きい方）。境界値 `current_x` は常に更新し、床未満の変化では再計算しません。  
また各外側ループの湿度連成前に、OFF / `current_x` 未初期化の空調は吸込湿度へ同期します（`current_x=0` 固定境界で室内が乾燥するのを防ぐ）。  
パススルー相当の還気循環は湿度移流から除外します。対象は OFF、および `COOLING` 以外の運転（暖房など）です。大風量還気を乾き／古い低湿度 BC のまま載せると外気湿度が薄まり、室湿度がほぼ 0 で固定されるためです。冷房 ON のみ還気枝＋吹出境界を残して除湿を反映します。

---

### 7. 能力上限の扱い

能力上限チェックは `checkAndAdjustCapacity()` で行います。

参照する上限:

- **`ac_spec.Q.<mode>.max`** を優先
- **`max` が無い場合は `ac_spec.Q.<mode>.mid`** を使用（DUCT_CENTRAL / LATENT_EVALUATE など `mid` のみの仕様に対応）
- 両方無い機種は能力制限を掛けません（上限なしとして扱う）

`Q.rtd` や `max_heat_capacity` は、この制御では参照しません。

---

### 8. DUCT_CENTRAL の処理熱量連動風量（外側ループ連成）

`model="DUCT_CENTRAL"` かつ運転中の機器については、ON/OFF が安定したあとに風量を見直します。

目的:

- 処理熱量と送風量の自己整合を取る
- 風量が変わることで換気/熱回路網の解が変わるため、outer loop 再計算につなぐ

仕様:

- 処理熱量が `0` のとき、目標風量は `0`
- 処理熱量が `Q.<mode>.rtd` のとき、目標風量は `V_inner.<mode>.dsgn`
- その間は線形補間（上限は `dsgn`）

式（mode は `heating` / `cooling`）:

- `ratio = clamp(Q_processed / (Q.<mode>.rtd * 1000), 0, 1)`
- `V_target = V_inner.<mode>.dsgn * ratio`

ここで:

- `Q_processed` は controller 内部の全熱（`Q_S + Q_L`, [W]）
- `Q.<mode>.rtd` は `ac_spec` 上の [kW]
- `V_inner.<mode>.dsgn` は [m3/s]

実装上は `in_node <-> aircon_node` の `fixed_flow` 換気枝を更新し、変更が入った場合は `shouldRecompute=true` を返します。

注意:

- この補正は DUCT_CENTRAL 専用です（他モデルには適用しません）。
- 現時点では `fixed_flow` 枝を対象とした運用を前提にしています。

---

### 9. 能力超過時の設定温度補正

エアコンが ON で、かつ

- `current totalCapacity > 最大能力（上記 max または mid）`

になった場合、`checkAndAdjustCapacity()` は**処理熱量が最大能力と等しくなる**設定温度を求め、`current_pre_temp` を補正します。

ここでの `totalCapacity` は **全熱（顕熱+潜熱）** [W] です。

方針:

- 暖房時: 設定温度を下げる
- 冷房時: 設定温度を上げる
- 運転モードは固定する
- ON/OFF は次の outer loop で再判定してよい

補正方法（2段階）:

```mermaid
flowchart TD
    OVER["全熱 > Qmax"] --> F["公式 findCapacityLimitedSetpoint"]
    F -->|有効| APPLY["effectiveSetpoint 更新<br/>再計算要求"]
    F -->|無効| BR["bracket 二分探索"]
    BR --> CH{"setpoint 変化?"}
    CH -->|Yes| APPLY
    CH -->|No・収束| DONE["bracket 消去 / Accept 候補"]
```

1. **公式による補正**  
   `findCapacityLimitedSetpoint(...)` で、入口温度・風量・運転モードを固定した近似のもと、`heatCapacity <= maxHeatCapacity` となる setpoint を算出。有効な setpoint が得られればそれを適用。
2. **二分探索（フォールバック）**  
   公式で有効解が得られない場合、熱ソルバの解（処理熱量）を利用した bracket 二分探索で、処理熱量 ≒ 最大能力 となる setpoint を求める。収束判定は「処理熱量が最大能力に十分近い」（相対 0.1% + 絶対 1W）で行い、bracket のみ狭まった場合はあと 1 回再計算してから完了。

重要: `stepCapacityLimitBracket()` は **現在点の処理熱量**で先に `capacityConverged` を判定する。収束していれば設定温度は動かさず bracket を消し、再計算もしない。未収束のときだけ bracket を更新して中点へ進む。

bracket 幅だけが `1e-3°C` 以下になった場合は、能力非超過側の端点を採用して `finalVerificationPending` を立て、**熱計算の最終確認をちょうど1回**行う。検証後に `Q <= Qmax + tol` なら終了。まだ超過なら bracket を拡張して探索再開し、拡張不能なら探索を打ち切って外側ループを止める（無限再計算防止）。

初期 bracket は ±10°C を起点に、顕熱近似で「能力以下の端」が見えるまで段階的に広げます（床 0°C / 天井 50°C）。

補正後に設定温度が動いた場合は `adjustmentMade=true` を返し、外側ループが同じ timestep を再計算します。  
処理熱量が最大能力を**下回る**状態で既に bracket が存在する場合（例: 設定を下げすぎて処理熱量が 0 に近い）は、設定温度を上げる方向に bracket を更新して探索を継続します。

### 9.1 要求設定と実効設定

| フィールド | 意味 |
|---|---|
| `current_requested_pre_temp` | スケジュールの要求設定温度 |
| `current_pre_temp` | 熱ソルバ fixed-row に使う実効設定（能力制限で動きうる） |
| `aircon_control_state` | `Off` / `SetpointControlled` / `CapacityLimited` |

- ON/OFF 判定は **要求設定** を参照する（deadband は維持）
- 能力制限は **実効設定** だけを動かす
- タイムステップ先頭で実効設定は要求設定へリセットされる

---

### 10. 現在の近似と制約

初期実装では、二分探索の**各試行ごとに full thermal solve はしていません**。

つまり、探索中は

- 入口温度
- 流量
- 運転モード

を固定した近似で setpoint を求め、最終的な整合は outer loop の再計算で取ります。

利点:

- 実装が小さい
- 既存の `shouldRecompute` 導線をそのまま使える
- 熱ソルバの係数行列再利用とも相性がよい

制約:

- 実際の連成系では setpoint と処理熱量の関係が完全単調とは限らない
- `AUTO` は `prepareRuntimeContext()` 内で暖房/冷房に解決された後のモードを固定して探索する
- より厳密な制御が必要なら、将来は「各試行ごとに熱計算も回す高精度版」へ拡張余地がある

---

### 11. ログ

能力チェック時は `solver.log` に次のような情報を出します。

- aircon key
- 最大処理熱量
- 現在処理熱量（全熱、顕熱/潜熱内訳付き）
- 超過/OK
- 補正前後の設定温度
- 再計算要求の有無

例（能力チェック）:

```text
ac1 最大処理熱量=500.00W (Q.heating.max 基準), 現在処理熱量(全熱)=820.00W (顕熱=700.00W, 潜熱=120.00W) → 超過, 設定温度補正=26.00→23.41°C, 再計算要求
ac1 最大処理熱量=500.00W (Q.heating.max 基準), 現在処理熱量(全熱)=499.20W (顕熱=430.00W, 潜熱=69.20W) → 二分探索収束 設定温度=23.10°C（処理熱量≒最大能力）
```

---

`bf` 選択時に RH>100% となった場合のログ例:

```text
[WARN] bf法の吹出点相対湿度が100%を超えたためRH95法へフォールバック: ac1 RH(bf)=102.31% -> RH(out)=95.00%
```

---

DUCT_CENTRAL 風量補正のログ例:

```text
ac1 DUCT_CENTRAL風量補正: 処理熱量=3624.00W, Q.rtd=7200.00W, 比率=0.5033, 風量 0.3000→0.1007 m3/s, 再計算要求
```

---

### 12. 今後の方針（段階導入）

現状は設定温度を動かして能力制限する近似を維持しつつ、次を段階導入します。

1. ~~二分探索収束時に設定温度が変わったのに再計算しない不整合の修正~~（済）
2. ~~`requested` / `effective` setpoint と `AirconControlState` の分離~~（済）
3. ~~`AirconStateProposal` + `AirconRecomputeReason` を外側ループへ配線~~（済: `runAirconIteration` が提案を集約し、メトリクスに `aircon_recompute_reasons_mask` を記録）
4. ~~`simulation_metrics` の outer 反復分布と `aircon_capacity_recalc` 割合を見られるようにする~~（済: mean/max/ge3_share/capacity_share）
5. 負荷が大きければ thermal solver に能力固定モードを追加し、`Qac=±Qmax` で室温を未知数に戻す（設定温度二分探索の廃止）

望ましい最終モデル:

| 状態 | 拘束 |
|---|---|
| Off | `Qac = 0` |
| SetpointControlled | `Troom = Trequested` かつ `|Qac| <= Qmax` |
| CapacityLimited | `Qac = ±Qmax`、室温は設定から外れうる |

---

### 13. 関連ドキュメント

- `docs/simulation_overview.md`
- `docs/acmodel_overview.md`
- `docs/aircon_spec_reference.md`（モデル別 ac_spec の形と能力上限キー）
- `docs/builder_json.md`

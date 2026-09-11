# 熱交換換気（顕熱・全熱）

`heat_recovery_vent` で、顕熱交換換気（HRV）と全熱交換換気（ERV）を設定できる。
エアコンのような温度制御ではなく、外気と還気のあいだで熱（と全熱時は湿気）を回収し、給気境界の温湿度を決める設備である。

対象はパッケージ v1.8.0、実装 commit `ef11982`。計算モデル全体との関係は技術文書v3.3第9章を参照する。

入力の厳密仕様は [`../engine/docs/builder_json.md`](../engine/docs/builder_json.md) の §11.0 を正本とする。

## 1. 顕熱と全熱のちがい

| `recovery` | 意味 | 給気温度 | 給気絶対湿度 |
|---|---|---|---|
| `sensible`（既定） | 顕熱交換 | \(\eta_t\) で回収 | 外気のまま（\(\eta_x=0\)） |
| `total` | 全熱交換 | \(\eta_t\) で回収 | \(\eta_x\) で回収 |

全熱は「顕熱＋潜熱」である。冬に室内の熱と水分を外気側へ移すと、給気は温められ湿度も上がる。夏はその逆になる。

別名: `sensible` ← `顕熱` / `hrv`、`total` ← `enthalpy` / `全熱` / `erv`。

## 2. ポート（OA / SA / RA / EA）

いずれも既存の `nodes[].key` を指す。

| ポート | 意味 | 個別キー | 短縮形 |
|---|---|---|---|
| OA | 外気取入 | `oa`（`outdoor`） | `outdoor` |
| SA | 給気先 | `sa`（`out`） | `room` |
| RA | 還気元 | `ra`（`in` / `return` / `room` / `set`） | `room` |
| EA | 排気先 | `ea`（`exhaust_out`） | `outdoor` |

- 短縮: `outdoor` + `room` → `oa=ea=outdoor`, `sa=ra=room`
- 個別: 給気と還気を別室・別チャンバにできる

展開後の空気経路:

```
OA ──► {key}（給気境界ノード, type=hrv） ──► SA
RA ──► {key}_exhaust（排気ジャンクション） ──► EA
```

## 3. 給気状態の式

圧力・熱の連成の前に、OA と RA の現在値から給気境界を更新する。

$$
T_{\mathrm{sa}} = T_{\mathrm{oa}} + \eta_t \bigl(T_{\mathrm{ra}} - T_{\mathrm{oa}}\bigr)
$$

$$
x_{\mathrm{sa}} =
\begin{cases}
x_{\mathrm{oa}} & (\texttt{sensible}) \\
x_{\mathrm{oa}} + \eta_x \bigl(x_{\mathrm{ra}} - x_{\mathrm{oa}}\bigr) & (\texttt{total})
\end{cases}
$$

- \(\eta_t\), \(\eta_x\) は 0..1（既定は \(\eta_t=0.7\)、全熱時 \(\eta_x=0.6\)）
- 顕熱では入力の \(\eta_x\) があっても 0 に固定する
- 給気ノードは `calc_t=false` / `calc_x=false` の固定境界。室の温度・湿度は移流で変わる

## 4. 回収熱量の出力

系列キーは機器の `key`。単位は W（[`units.md`](units.md)）。

| 系列 | 内容 |
|---|---|
| `hrv_sensible_heat` | 回収顕熱 \(\rho c_p Q_{\mathrm{eff}}(T_{\mathrm{sa}}-T_{\mathrm{oa}})\) |
| `hrv_latent_heat` | 回収潜熱 \(\rho L Q_{\mathrm{eff}}(x_{\mathrm{sa}}-x_{\mathrm{oa}})\)（顕熱時は 0） |

有効風量は \(Q_{\mathrm{eff}}=\min(Q_{\mathrm{sa}}, Q_{\mathrm{ea}})\)。給気・排気の実風量がずれる場合（ファン PQ や圧損の非対称）は小さい方を使う。

正の値は「外気を室内寄りに近づけた」方向の回収（暖房期の予熱・加湿など）に対応する。冷房期は符号が反転しうる。

## 5. 入力例

### 5.1 全熱・固定風量（同一室）

```python
"heat_recovery_vent": [{
    "key": "ERV1",
    "outdoor": "外部",
    "room": "LD",
    "recovery": "total",
    "eta_t": 0.75,
    "eta_x": 0.65,
    "vol": 150 / 3600,  # m³/s
}]
```

湿度を解く場合は室側に `calc_x=true`（または湿気容量展開）が必要である。全熱回収は給気の \(x\) 境界を動かすため、湿度計算 OFF だと潜熱側の効果は室へ伝わらない。

### 5.2 全熱・ポート分離

```python
"heat_recovery_vent": [{
    "key": "ERV1",
    "oa": "外部",
    "sa": "給気チャンバ",
    "ra": "還気チャンバ",
    "ea": "排気口",
    "recovery": "total",
    "eta_t": 0.8,
    "eta_x": 0.5,
    "vol": 200 / 3600,
}]
```

### 5.3 顕熱・給気／排気ファン PQ

```python
"heat_recovery_vent": [{
    "key": "HRV1",
    "oa": "外部",
    "sa": "廊下",
    "ra": "LD",
    "ea": "外部",
    "recovery": "sensible",
    "eta_t": 0.7,
    "supply":  {"p_max": 100.0, "p1": 50.0, "q1": 0.05, "q_max": 0.12},
    "exhaust": {"p_max": 90.0,  "p1": 45.0, "q1": 0.04, "q_max": 0.11},
    "area": 0.05,
    "k_total": 1.0,
}]
```

`supply` / `exhaust` はファン PQ 用であり、ポート名ではない。片方だけ指定するとエラーになる。PQ 時は関連ノードの圧力を未知数として解く。

## 6. エアコンとの関係・制約

- 熱交換換気は ON/OFF・設定温度・能力制限を持たない。常時、効率一定の受動回収である
- バイパス、着霜防止、ファン電力、効率の温湿度依存は未実装
- ダクト中央空調の `V_vent`（ファン電力用）とは別物である
- 実装: builder `engine/app/builder/heat_recovery_vent.py`、solver `engine/solver/hrv/`

## 7. 関連文書

| 文書 | 内容 |
|---|---|
| [`../engine/docs/builder_json.md`](../engine/docs/builder_json.md) §11.0 | 入力フィールド正本 |
| [`builder_input_quickstart.md`](builder_input_quickstart.md) | 入力の組み立て |
| [`units.md`](units.md) | 出力単位 |
| [`aircon_humidity_control.md`](aircon_humidity_control.md) | 空調の湿気境界（別設備） |

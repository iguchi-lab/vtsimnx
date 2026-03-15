# エアコン仕様（ac_spec）リファレンス

このドキュメントは、エアコンモデル別の **ac_spec（仕様JSON）** の形と、solver が参照する **能力上限キー** を一覧にしたものです。

- 制御の流れ・能力超過時の補正: `docs/aircon_control_overview.md`
- 各モデルの式・入力出力: `docs/acmodel_overview.md`
- builder 入力での空調設定: `docs/builder_json.md` の「aircon」節

---

## 1. 共通事項

### 1.1 モデル指定

| 指定元 | キー | 説明 | デフォルト |
|--------|------|------|------------|
| builder 入力 | `aircon.model` | COP/電力推定に使うモデル種別 | 未指定時 **RAC** |
| solver 入力 | `nodes[].model` | 同上（parser が未指定時 **RAC** を適用） | - |

有効な値: `CRIEPI`, `RAC`, `DUCT_CENTRAL`, `LATENT_EVALUATE`

### 1.2 能力上限の参照（solver 側）

能力超過チェック（`checkAndAdjustCapacity`）で使う「最大処理熱量」は、次の優先順で取得します。

1. **`Q.<mode>.max`**（cooling / heating ごと） [kW]
2. **`Q.<mode>.max` が無い場合** → **`Q.<mode>.mid`** [kW]

どちらも無い機種は能力制限を掛けません（上限なし）。

### 1.3 潜熱計算方式（`latent_method`）

`ac_spec` では潜熱処理の方式を次で指定できます。

- `latent_method: "rh95"`（**デフォルト**）  
  吹出温度 `Tout` から吹出空気 RH を 95% として `supplyX` を決める
- `latent_method: "bf"`  
  バイパスファクタ法で `supplyX` を計算  
  BF は `bf` / `BF` / `bypass_factor` で指定（省略時 `0.2`、内部で `0.0..0.99` にクランプ）。  
  計算結果の吹出 RH が 100% 超なら警告を出し、`rh95` へフォールバック。
- `latent_method: "coil_aoaf"`（別名: `"aoaf"`, `"literature"`）  
  コイル前面風速・有効表面積を用いる文献式（4.2.1）に基づく潜熱評価。
  - 追加パラメータ:
    - `Af` または `coil_face_area` : 実質コイル前面面積 [m²]（既定 `0.133`）
    - `Ao` または `coil_surface_area` : コイル有効表面積 [m²]（既定 `4.84`）
  - 顕熱処理量 `Hs` [W] と吸込/吹出条件から `Hr` [W] を評価し、`Q_L` および `supplyX` を決定する
- `latent_method: "none"`  
  潜熱処理なし（`Q_L=0`、`supplyX = X_in`）

補足:

- 顕熱・潜熱とも単位は [W]
- 能力上限判定は **全熱（`Q_S + Q_L`）** で実施

---

## 2. モデル別 ac_spec の形

### 2.1 CRIEPI

- **必須**: `Q`, `P` の cooling/heating × **min, rtd, max**
- **必須**: `V_inner`, `V_outer` の cooling/heating × **rtd**（風量 [m³/s]）
- 単位: `Q` / `P` は [kW]、風量は [m³/s]

能力上限: **`Q.<mode>.max`** を推奨（`mid` は CRIEPI では通常使わない）。

最小形の例は `docs/acmodel_overview.md` の「CRIEPI向けの例」を参照。

---

### 2.2 RAC

- **必須**: `Q`, `P` の cooling/heating × **rtd, max**
- **任意**: `dualcompressor`（bool、容量可変型の切り替え）
- 単位: `Q` / `P` は [kW]

能力上限: **`Q.<mode>.max`** を指定するのが一般的。

---

### 2.3 DUCT_CENTRAL

- **必須**: `Q`, `P` の cooling/heating × **rtd, mid, min** 等
- **必須**: `P_fan`, `V_inner`（必要に応じ `V_outer`）の cooling/heating × **rtd, mid, dsgn** 等
- 単位: `Q` / `P` / `P_fan` は [kW]、風量は [m³/s]
- 運転点入力 `InputData` では `V_vent`（換気分, [m³/s]）を使用可能。`V_outer` とは別入力で、既定は `0`。

能力上限: **`Q.<mode>.max`** があればそれを使用。無い場合は **`Q.<mode>.mid`** を能力上限として使用。

---

### 2.4 LATENT_EVALUATE

- **必須**: `Q`, `P` の cooling/heating × **rtd, mid, min** 等
- **必須**: `P_fan`, `V_inner` の cooling/heating × **rtd, mid, dsgn** 等
- 単位: 上記と同様 [kW], [m³/s]

能力上限: **`Q.<mode>.max`** または **`Q.<mode>.mid`**（仕様に合わせてどちらか一方でよい）。

---

## 3. 一覧表（能力上限キー）

| モデル | 能力上限に使うキー | 備考 |
|--------|--------------------|------|
| CRIEPI | `max` | min/rtd/max を揃えて指定する想定 |
| RAC | `max` | rtd + max が一般的 |
| DUCT_CENTRAL | `max` または `mid` | mid のみの仕様でも制限がかかる |
| LATENT_EVALUATE | `max` または `mid` | 同上 |

---

## 4. 関連ドキュメント

- `docs/aircon_control_overview.md` — エアコン制御・能力超過時の補正
- `docs/acmodel_overview.md` — 各モデルの式・入力出力・参考文献
- `docs/builder_json.md` — builder の `aircon` と `ac_spec` の渡し方

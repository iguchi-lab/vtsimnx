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

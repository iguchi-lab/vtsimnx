### エアコンモデル（acmodel）の概要

このドキュメントは、C++ の `acmodel`（`/acmodel`）が提供する **エアコンCOP/電力推定モデル**について、入力（`ac_spec` / 運転点入力）と出力、solver との接続をまとめたものです。

---

### 1. モデルの種類

`acmodel::AirconModelFactory::createModel(typeStr, spec)` で選択します。

- `CRIEPI`（`CRIEPIModel`）: CRIEPI系の係数モデル
- `RAC`（`RACModel`）: ルームエアコンモデル（顕熱/潜熱、着霜補正等）
- `DUCT_CENTRAL`（`DuctCentralModel`）:（将来拡張）
- `LATENT_EVALUATE`（`LatentEvaluateModel`）:（将来拡張）

---

### 2. solver から acmodel に渡す入力（運転点）

solver 側は、各タイムステップのエアコン運転点を `acmodel::InputData` として渡します。

- `T_in` / `T_ex` : 室内/外気温度 \([°C]\)
- `X_in` / `X_ex` : 室内/外気絶対湿度 \([kg/kg(DA)]\)
- `Q` : 要求能力 \([W]\)（現状は顕熱要求を主に入れる）
- `Q_S` / `Q_L` : 顕熱/潜熱 \([W]\)（潜熱は今後拡張）
- `V_inner` / `V_outer` : 内外風量 \([m³/s]\)

#### 湿度入力の扱い（重要）

- `calc_flag.x=true` の場合、solver は **ネットワーク上の `current_x`** を優先して `X_in/X_ex` に入れます。
- 外気ノード等で `x` が取得できない場合は、警告のうえ **JIS固定条件へフォールバック**します（互換挙動）。

---

### 3. ac_spec（仕様JSON）の最低限の形

全モデル共通で、基本的に `Q`（能力）と `P`（消費電力）が必要です。

#### 3.1 共通：`Q` と `P`

- `Q`: `{ "cooling": {...}, "heating": {...} }`
- `P`: `{ "cooling": {...}, "heating": {...} }`

内部では `min/rtd/max` のようなキー（例）を参照します（モデルにより利用範囲は異なります）。

#### 3.2 CRIEPI: 風量

CRIEPI 系では、仕様として以下を使います（代表）:

- `V_inner`: `{ "cooling": {"rtd": ...}, "heating": {"rtd": ...} }`
- `V_outer`: `{ "cooling": {"rtd": ...}, "heating": {"rtd": ...} }`

※ `Pc`（定数消費電力）や多項式係数は、初期化時に `ac_spec` から計算され、`getModelParameters()` で参照できます。

最小例（CRIEPI向けの例）:

```json
{
  "Q": {
    "cooling": {"min": 0.7, "rtd": 2.2, "max": 3.3},
    "heating": {"min": 0.7, "rtd": 2.5, "max": 5.4}
  },
  "P": {
    "cooling": {"min": 0.095, "rtd": 0.395, "max": 0.780},
    "heating": {"min": 0.095, "rtd": 0.390, "max": 1.360}
  },
  "V_inner": {"cooling": {"rtd": 0.2016667}, "heating": {"rtd": 0.2183333}},
  "V_outer": {"cooling": {"rtd": 0.4700000}, "heating": {"rtd": 0.4250000}}
}
```

#### 3.2.1 CRIEPIモデル（論文の考え方：冷房運転）

`CRIEPIModel` の背景として、CRIEPI（電力中央研究所）による「家庭用エアコンの熱源特性モデル」（冷房運転）があります。
本モデルは、カタログ（仕様書）情報から機器固有パラメータを同定し、任意の室内外条件・負荷条件で COP/電力を推定することを目的としています（詳細は論文参照）。

- **基本関係**:
  - 成績係数: \(COP = Q/P\)
  - 消費電力の分解: \(P = P_{comp} + P_{aux}\)（圧縮機＋補機/ファン等）
- **モデルの骨格（概念）**:
  - 冷凍サイクルの「理論成績係数」と「実成績係数」の比（効率）を導入し、運転条件（室内外温湿度、風量、負荷）から COP/電力を推定する
  - 仕様書の公表値（能力・消費電力・風量など）から、機器固有の定数項（補機電力相当）や係数を決める

参考（CRIEPI論文）: [家庭用エアコンの熱源特性モデルの開発（その1：冷房運転時モデル）](https://doi.org/10.18948/shase.38.190_41)

#### 3.3 RAC: 追加仕様（例）

RAC は Python版互換の実装で、例えば以下のフラグを参照します:

- `dualcompressor`（bool）

また、冷房/暖房で外気条件（温度・湿度）が式に入ります。特に **外気絶対湿度 `X_ex`** を運転点入力として必要とします。

最小例（RAC向けの例）:

```json
{
  "Q": {
    "cooling": {"rtd": 2.2, "max": 2.8},
    "heating": {"rtd": 2.2, "max": 3.6}
  },
  "P": {
    "cooling": {"rtd": 0.455, "max": 0.745},
    "heating": {"rtd": 0.385, "max": 1.070}
  },
  "dualcompressor": false
}
```

#### 3.3.1 RACモデル（資料の考え方：直吹き壁掛けルームエアコン）

`RACModel` は、住宅用の「直吹き形かつ壁掛け形（家庭用）」ルームエアコンを対象とした計算手法（章4-3）を背景に、外気条件・部分負荷・補正係数等を考慮して、消費電力や最大出力を推定します（詳細は資料参照）。

- **適用範囲（概念）**:
  - 家庭用の直吹き壁掛け形（マルチタイプは対象外）
- **補正の考え方（例）**:
  - 室内機吹出風量に関する補正係数 \(C_{af}\)
  - デフロストに関する補正係数 \(C_{df}\)（主に暖房側）
  - 室内機吸込み湿度に関する補正係数 \(C_{hm}\)
- **単位系の注意**:
  - 仕様の「能力」は \(q_{rtd}\,[W]\) が基本だが、途中の計算では負荷/処理量を \(MJ/h\) で扱う式があり、換算（\(W \leftrightarrow MJ/h\)）が前提になる

参考（RACモデル資料）: [第四章 暖冷房設備／第三節 ルームエアコンディショナー（Ver.08, 2025.04）](https://www.kenken.go.jp/becc/documents/house/4-3_250401_v08.pdf)

---

### 4. 出力

`estimateCOP(mode, input)` の戻り値は `acmodel::COPResult` です。

- `COP` : 成績係数 \([-]\)
- `power` : 消費電力 \([kW]\)
- `valid` : 計算が有効か
- `logMessages` : 詳細ログ（verbosityに応じて solver が出力）

solver 側は `power` を \([W]\) に変換して出力します。

単位の注意:

- `ac_spec` の `Q`/`P` はモデル実装により **kW前提の項目がある**ため、入力は既存仕様に揃えてください（solver側のエアコン制御は W で扱い、acmodel内部で必要に応じて換算します）。

---

### 5. ログ

acmodel は `acmodel::setLogger()` と `acmodel::setLogVerbosity()` でログ制御します。
solver 側で `verbosity>=2` のとき、モデル内部の詳細ログを出力します。

---

### 6. 関連ドキュメント

- builder 入力: `docs/builder_json.md`
- シミュレーション全体: `docs/simulation_overview.md`
- 参考文献（CRIEPI冷房モデル）: [家庭用エアコンの熱源特性モデルの開発（その1）](https://doi.org/10.18948/shase.38.190_41)
- 参考文献（RACモデル）: [第四章 暖冷房設備／第三節 ルームエアコンディショナー（Ver.08）](https://www.kenken.go.jp/becc/documents/house/4-3_250401_v08.pdf)



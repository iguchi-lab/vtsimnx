### 応答係数法（CTF / response_conduction）入力仕様（VTSimNX）

このドキュメントは、熱計算で **応答係数法（CTF: Conduction Transfer Function / Response Factor）**を使うための入力 JSON の書き方と、係数の単位・符号の約束をまとめたものです。

---

### 1. 何が変わるか（RCとの違い）

- **RC法**: 壁を層ノードに分割して、内部層ノード（熱容量）をネットワークに持つ。
- **応答係数法（CTF）**: 壁内部の状態は「係数＋履歴」で表現し、ネットワーク上は **両端表面ノード2つ＋`response_conduction`ブランチ1本**のみ。
  - 壁体内の蓄熱を持つため、**両面の熱流は一般に一致しません**（表面別に `q_src`, `q_tgt` を扱う）。

RC法の入力仕様は以下にまとめています:

- `docs/thermal_rc.md`

---

### 2. 単位の約束（重要）

この実装は **係数を「面積あたり（per m²）」**として扱います（ユーザー選択: option B）。

- **温度**: \([^\circ C]\) または \([K]\)（差分のみ効くのでどちらでも同じ）
- **熱流**: solver内部の熱収支は \([W]\)
- **CTF係数**: **熱流密度** \(q''\) を出す係数として扱う
  - \(q''\) の単位: \([W/m^2]\)
  - `resp_a_*`, `resp_b_*`: \([W/m^2/K]\)
  - `resp_c_*`: 無次元（過去の \(q''\) に掛かる係数）
  - solver が温度依存項に **面積 \(A\)** を掛けて \([W]\) に変換します

そのため **`response_conduction` では `area` が必須**です（validation でチェックします）。

---

### 3. surface での指定（推奨：builder に任せる）

#### 3.1 全サーフェス一括で response にする（JSONで指定）

```json
{
  "builder": {
    "surface_layer_method": "response"
  }
}
```

#### 3.2 surface個別に response を指定

```json
{
  "surfaces": [
    {
      "key": "室->外部",
      "part": "wall",
      "area": 10.0,
      "layer_method": "response",
      "layers": [
        {"lambda": 0.8, "t": 0.12, "v_capa": 900000.0}
      ]
    }
  ]
}
```

`layer_method="response"` かつ `layers` がある場合、builder は以下を生成します:

- 表面ノード2つ（室側/外側）
- 対流ブランチ2本（室ノード→室側表面, 外側表面→外気ノード）
- `response_conduction` ブランチ1本（室側表面↔外側表面）

---

### 4. 係数の指定（手入力 or 自動生成）

#### 4.1 手入力（surface["response"] を指定）

`surface.response` を与えると、builder はその係数を使って `response_conduction` を生成します。

```json
{
  "surfaces": [
    {
      "key": "A->B",
      "part": "wall",
      "area": 10.0,
      "layer_method": "response",
      "layers": [{"lambda": 1.0, "t": 0.1, "v_capa": 1000.0}],
      "response": {
        "resp_a_src": [5.0],
        "resp_b_src": [-5.0],
        "resp_a_tgt": [5.0],
        "resp_b_tgt": [-5.0],
        "resp_c_src": [],
        "resp_c_tgt": []
      }
    }
  ]
}
```

#### 4.2 自動生成（surface["response"] を省略）

`layer_method="response"` で `layers` があり、かつ `response` が無い場合、builder が `time_step` と物性（`lambda`, `t`, `v_capa`）から係数を自動生成します。

自動生成の方法と項数（次数）は `builder` で指定できます:

```json
{
  "builder": {
    "surface_layer_method": "response",
    "response_method": "modal_expsum",
    "response_terms": 6
  }
}
```

- `response_method`:
  - `"arx_rc"`（既定）: RC連鎖（状態数＝層数）をそのままARX化（次数＝層数）
  - `"modal_expsum"`: 離散系をモード分解して **指数項（λ^k）の和**で近似し、`response_terms` 個に縮約してARX化
- `response_terms`:
  - `"modal_expsum"` のときに効く「項数（次数）」です（未指定なら層数相当）
  - `resp_a_* / resp_b_*` の長さは `response_terms+1`
  - `resp_c_*` の長さは `response_terms`

---

### 5. 安定化フォールバック（sum(c) が 1 に近い場合）

地中・超厚壁など「非常に遅い系」では、AR係数 `resp_c_*` の和が 1 に極端に近づき、丸め誤差で不安定になりやすいことがあります。

その場合 builder は、その壁に限って **定常U値（メモリなし）**へフォールバックすることがあります。

- `resp_c_*` は空
- `resp_a_*=[U]`, `resp_b_*=[-U]`

---

### 6. 符号（熱流の向き）

`response_conduction` は src側・tgt側それぞれの「壁体へ流入する向き」を正として熱流密度を計算します。
（この向きの約束に合わせて solver 側の符号が実装されています）



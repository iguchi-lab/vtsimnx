# `vt.run_calc` 入力JSONクイックスタート（ユーザー向け）

このページは、`vtsimnx/vtsimnx` の利用者が  
**`vt.run_calc(base_url, input_data)` に渡す `input_data` をどう作るか**を最短で把握するためのガイドです。

- 対象: Python から `vtsimnx` を呼ぶ利用者
- ゴール: engine 内部設定ではなく、**利用者が書く JSON（dict）** の構造を理解する

---

## 1. まず押さえる前提

- 計算実行は `vt.run_calc(...)` で行います。
- 入力は Python の `dict`（JSON互換）で組み立て、`config_json` として渡します。
- `surfaces`, `aircon`, `heat_source` などを含むと、builder が展開して solver 入力を構成します。
- 利用者は通常、engine 側コードを編集せずに `input_data` 側を調整して運用します。

---

## 2. 最小入力テンプレート

```python
import vtsimnx as vt

input_data = {
    "builder": {},
    "simulation": {
        "index": {
            "start": "2026-01-01 01:00:00",
            "end": "2026-01-02 00:00:00",
            "timestep": 3600,
            "length": 24,
        }
    },
    "nodes": [
        {"key": "外部", "t": 5.0},
        {"key": "室1", "calc_t": True, "v": 30.0},
    ],
    "ventilation_branches": [
        {"key": "外部->室1", "source": "外部", "target": "室1", "type": "fixed_flow", "vol": 30.0},
    ],
    "thermal_branches": [
        {"key": "外部->室1", "source": "外部", "target": "室1", "type": "conductance", "conductance": 50.0},
    ],
}

result = vt.run_calc("http://127.0.0.1:8000", input_data)
```

---

## 3. よく使うキーの考え方（利用者目線）

- `builder`
  - builder の挙動切り替え。通常は空 dict（`{}`）で開始し、必要時のみフラグを追加します。
- `simulation.index`
  - 解析期間。配列時系列を使う場合の基準長になります。
- `nodes`
  - 状態点（室、外部、容量、空調など）。`key` は他セクションから参照される識別子です。
- `ventilation_branches`
  - 空気の接続（風量・圧力系）。
- `thermal_branches`
  - 熱の接続（コンダクタンス・発熱など）。
- `surfaces`
  - 壁/床/窓などの面要素を定義し、builder による展開を使う場合に指定します。
- `aircon`, `heat_source`
  - 空調制御と内部発熱を入れる場合に追加します。

---

## 4. 時系列の書き方

- 多くの値は `number`（定数）または `number[]`（時系列）で指定可能です。
- 時系列配列長は原則 `simulation.index.length` に合わせます。
- `vt.run_calc` は `pandas.Series` / `DatetimeIndex` も受け取れるため、前処理で `DataFrame` を作る運用と相性が良いです。

---

## 5. まずはこの順で組み立てる

1. `simulation.index` を固定する  
2. `nodes` の `key` を確定する  
3. `ventilation_branches` / `thermal_branches` の `source` / `target` を接続する  
4. 必要なら `surfaces` を追加する  
5. 最後に `aircon` / `heat_source` を追加する  

この順にすると、参照切れ（存在しないノード名）を早い段階で潰せます。

---

## 6. 典型ミス

- `source` / `target` に存在しない `nodes[].key` を指定する
- 配列長が `simulation.index.length` と一致しない
- 単位が混在する（例: 風量を m3/s と m3/h で混在）
- 最初から全機能を入れて切り分け不能になる

まずは最小ケースで `vt.run_calc` を通し、要素を段階的に増やす運用を推奨します。

---

## 7. 実行手順（ローカル）

1. APIサーバーを起動する（`engine/RUN_FASTAPI.md` 参照）  
2. API URL を設定する  
   - `export VTSIMNX_API_URL=http://127.0.0.1:8000`
3. 最小サンプルを実行する  
   - `python ../examples/run_calc_minimal.py`

---

## 8. よくあるエラーと対処

- `ConnectionError` / `ReadTimeout`
  - APIサーバー未起動、URL誤り、ポート未公開を確認してください。
- `source` / `target` の参照エラー
  - `nodes[].key` と完全一致しているかを確認してください（全角半角・大小文字含む）。
- 時系列長の不一致
  - 配列の要素数を `simulation.index.length` に揃えてください。
- まずどこから確認するか分からない
  - `request_output_path` で送信JSONを保存し、最小ケースとの差分を見るのが最短です。

---

## 9. 関連ドキュメント

- 全体導線: `README.md`
- 実サンプルコード: `../examples/README.md`, `../examples/run_calc_minimal.py`, `../examples/vs_simheat_sample.py`
- ノード/ブランチ早見表: `node_branch_schema.md`
- `surfaces` の実務ガイド: `surface_usage.md`
- スケジュール作成: `schedule_usage.md`

### 空調を使う場合の注意（DUCT_CENTRAL）

- DUCT_CENTRAL モデルでは、solver が処理熱量に応じて送風量を再評価します。
- 目安は `Q=0 -> V=0`, `Q=Q.rtd -> V=V_inner.dsgn`（中間は線形）です。
- 送風量変更で換気・熱連成の解が変わるため、同じ timestep 内で再計算が走る場合があります。
- 実装詳細は `../engine/docs/aircon_control_overview.md` と `../engine/docs/acmodel_overview.md` を参照してください。

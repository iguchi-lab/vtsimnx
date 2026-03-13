# builder 配下の整理メモ

surfaces リファクタ後、他モジュールで検討・対応するとよい点をまとめたものです。

---

## 1. 重複の解消（推奨）

### 1.1 `_scalar_initial_temperature`

- **surfaces.py**: ノードの `t`（スカラー or 時系列）から初期値を取り出す。`list, tuple, np.ndarray` に対応。
- **thermal.py**: ほぼ同じ処理。`list, tuple` のみ（`np.ndarray` 未対応）。

**案**: thermal から surfaces の `_scalar_initial_temperature` を流用する（`from .surfaces import _scalar_initial_temperature`）。循環 import なし。

### 1.2 `_series_summary`（ログ用時系列サマリ）

- **heat_sources.py**: `_series_summary(v)` — 時系列の len / 先頭・末尾を短く表示。
- **moisture.py**: 同一名・ほぼ同一実装。

**案**: `utils.py` に `series_summary_for_log(value) -> str` を追加し、heat_sources と moisture の両方で利用する。

---

## 2. builder.py のオプション解決

`_resolve_builder_options` は次のパターンが多数繰り返されている。

- `if add_XXX is None: add_XXX = _pick_bool(builder_opt, "add_XXX")`
- トップレベルでの上書き
- 最終デフォルト `add_XXX = True if add_XXX is None else bool(add_XXX)`

**案**: オプション名のリストでループする、または「オプション名 → デフォルト値」の dict で一括解決するヘルパーにすると、追加・変更がしやすくなる。既存の `surface_layer_method` / `response_method` / `response_terms` は別扱いのまま。

---

## 3. heat_sources.py の放射分

`build_heat_generation_branches` 内の「放射分: 対象室の表面へ面積按分」ブロックが長い。

**案**: `_append_radiation_branches(branches, room, q_rad, surfaces, key)` のようなヘルパーに切り出し、対流分・放射分の流れが読みやすくなる。

---

## 4. validate.py

- 約 930 行。`validate_node_config` / `validate_ventilation_config` / `validate_thermal_config` などセクション別に分かれている。
- 現状のままでも役割は明確。無理にファイル分割しなくてよい。
- 必要になったら「ノード検証」「換気検証」「熱検証」で別モジュールに分離する選択肢あり。

---

## 5. その他

- **parsers.py**: `_parse_*` が並んでおり、役割ごとに分かれていて読みやすい。
- **aircon.py**: 短く、処理も単純。特段の整理は不要。
- **config_types.py / logger.py**: 現状で問題なし。

---

## 実施済み（本メモ作成時に反映）

- **utils.py**: `series_summary_for_log` を追加。
- **heat_sources.py / moisture.py**: 上記を利用するよう変更。
- **thermal.py**: `_scalar_initial_temperature` を surfaces から import するよう変更。

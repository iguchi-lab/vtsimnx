# 公開 API の安定性

`vtsimnx` ルートから再エクスポートする記号の安定性区分です。  
機械可読な一覧は `vtsimnx.api_stability`（`STABLE` / `EXPERIMENTAL` / `DEPRECATED`）です。

## 区分

| 区分 | 意味 | 互換保証 |
|---|---|---|
| **stable** | 推奨の公開面 | MINOR/PATCH で破壊しない。破壊は MAJOR |
| **experimental** | 便利だが契約未固定 | 予告なく変更しうる |
| **deprecated** | 廃止予定 | `remove_in` まで残し、アクセス時に `DeprecationWarning` |

## stable

| 記号 | 戻り値 / 備考 |
|---|---|
| `run_calc` | `CalcRunResult` |
| `CalcRunResult` | 実行結果コンテナ |
| `RunCalcAPIError` | API エラー |
| `get_artifact_file` | `Path` または bytes |
| `__version__` / `get_version` | `str`（正本は `pyproject.toml`） |
| `units` | 単位定数モジュール |

推奨 import:

```python
import vtsimnx as vt

result = vt.run_calc(...)
print(vt.get_version())
```

## experimental

日射・快適性・I/O・材料・スケジュールなど。トップレベル再エクスポートは互換のため残していますが、  
本番コードでは可能な限りサブモジュールから import してください。

例: `from vtsimnx.archenv import solar_gain_by_angles`

## deprecated（廃止予定）

トップレベル名は `2.0.0` で削除予定です。代替は `vtsimnx.schedule` です。

| 旧（トップレベル） | 代替 | 削除予定 |
|---|---|---|
| `make_8760_data` | `vtsimnx.schedule.make_8760_data` | 2.0.0 |
| `ac_mode` | `vtsimnx.schedule.ac_mode` | 2.0.0 |
| `pre_tmp` | `vtsimnx.schedule.pre_tmp` | 2.0.0 |
| `pre_rh` | `vtsimnx.schedule.pre_rh` | 2.0.0 |
| `vol` | `vtsimnx.schedule.vol` | 2.0.0 |
| `sensible_heat` | `vtsimnx.schedule.sensible_heat` | 2.0.0 |

```python
# NG（警告）
from vtsimnx import make_8760_data

# OK
from vtsimnx.schedule import make_8760_data
```

## 単位

温度・圧力・流量・熱量などの単位は `docs/units.md` および `vtsimnx.units` を参照してください。  
OpenAPI / Pydantic スキーマでは `json_schema_extra.unit` に同じ表記を載せます。

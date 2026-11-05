# 入力 JSON の雛形

[入力JSONの組み立て](../../docs/ai_input_guide.md) の用途別例です。`vt.run_calc` の実行コードは [`../run_calc_minimal.py`](../run_calc_minimal.py) です。

| ファイル | 足すもの |
|---|---|
| `minimal_room.json` | 一室・給排気バランス |
| `heat_source.json` | 発熱 [W] |
| `humidity_source.json` | 発湿 [kg/s] |
| `surfaces_wall.json` | 壁面 |
| `aircon_rac.json` | RAC 空調 |

風量 `vol` は m³/s です（30 m³/h = 0.008333...）。

# 入力面から住宅の UA・ηAC・ηAH を求める

室温計算に渡す `input_data["surfaces"]` を使い、気象系列やソルバーの実行前に外皮を評価できます。標準住宅では床断熱の `default`（等級4）・`grade5`～`grade7`、基礎断熱の `foundation4`～`foundation7` に対応します。

```python
import json
from pathlib import Path
import vtsimnx as vt

# 初回に、make_caseで選んだ住宅と同じモデルの評価条件を書き出す。
path = Path("foundation5-rating.json")
if not path.exists():
    template = vt.standard_house.rating_template("foundation5", region=6)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
```

JSONの `null` に設計値を記入し、次のセルで集計します。

```python
assessment = json.loads(path.read_text(encoding="utf-8"))
r = vt.envelope_performance(input_data, assessment)
print(r["UA"], r["eta_AC"], r["eta_AH"])
import pandas as pd
display(pd.DataFrame(r["parts"]))
```

- **窓・ドア**　`U` は枠と表面熱抵抗を含む開口部全体の熱貫流率［W/(m²·K)］です。等級5～7にはモデル元のSimHeat部位表の値が入ります。障子などの付属部材の扱いを含め、採用する評価仕様と照合してください。等級4には評価用の値を記入します。窓の `eta_d` は枠込みの垂直面日射熱取得率（0～1）です。標準住宅の既定は取得日射熱補正係数の定数法（冷房期0.93、暖房期0.51）。別の方法で評価した窓は `f_C`・`f_H` を上書きします。SimHeatのカーテン込みの放射・対流成分や時刻別の日向率とは定義が異なります。
- **床・基礎**　床断熱では通気する床裏に面する床を H=0.7 とし、基礎断熱では床下を断熱境界内に含めます。土間は面積を分母に加算し、熱損失は `lines` に区間別の中心線長さ［m］、`psi`［W/(m·K)］、温度差係数 `H` を入力します。標準住宅は基礎壁を面として評価する方式です。これに対応する土間外周ψを選んでください。玄関土間と床下の間など、隣接条件が異なる区間は行を分けます。
- **一般部位**　`parts` の各項目は同じ部位の並列断面を `keys` で参照します。面積比でUを加重し、面積は断面を合算して0.01 m²単位に丸めます。層構成は室内側から並べ、熱抵抗は厚さ［m］÷熱伝導率または明示した `thermal_resistance` を使います。外気に連通する通気層で加算を止めます。基準用の熱抵抗・施工補正がシミュレーションと異なる部位には評価用の `U` を明示してください。外壁等の日よけ効果係数 `gamma_C`・`gamma_H` は既定1、日射吸収率 `absorptance` は既定0.8です。
- **任意の住宅**　`assessment` はモデル本体と独立した辞書です。全入力面を `parts` か `excluded` に一度ずつ割り当てます。`kind` は wall、roof、ceiling、floor、window、door、slab。層からUを求める面に外気側表面抵抗の区分 `outside`（真偽値）を設定します。`directions` は外気に面する方位（N～NW、屋根はTop、下面はBottom）のリスト、日射の当たらない隣接空間は空リストです。線熱橋は `kind="bridge"` とし、複数方位の平均では日射の当たらない側を `None`（JSONのnull）とします。土間外周は `kind="perimeter"` です。
- **結果**　UAは小数第2位に切り上げ、ηACは小数第1位に切り上げ、ηAHは小数第1位に切り下げます。8地域のηAHは対象外です。`A_env`［m²］、`q`［W/K］、`m_C`・`m_H`［m²］と部位別内訳も返します。

計算根拠は建築研究所の[第三章第二節「外皮性能」](https://www.kenken.go.jp/becc/documents/house/3-2_260401_v13.pdf)、[第三節「熱貫流率及び線熱貫流率」](https://www.kenken.go.jp/becc/documents/house/3-3_261116_v28.pdf)、[第四節「日射熱取得率」](https://www.kenken.go.jp/becc/documents/house/3-4_261001_v21.pdf)です。

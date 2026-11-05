# モデル作成を AI に頼むとき

> **使い方** — 人間が GitHub の URL を AI に渡して入力を作らせるときの指令です。入力の意味は [入力JSONの組み立て](ai_input_guide.md)。住宅1棟の層構造は [`examples/vs_simheat_sample.py`](../examples/vs_simheat_sample.py) と [SimHeat 比較ケース](vs_simheat_example.md)。

このページを先に読んでから、入力を書いてください。展開済みの巨大 JSON を一枚で書かないでください。

## 1. どの例を真似るか

| 頼まれたもの | 真似る場所 | 出力の形 |
|---|---|---|
| 一室、発熱、発湿、壁1面、RAC 1台 | [`examples/ai_input/`](../examples/ai_input/) | 短い JSON（または同じ内容の `dict`） |
| 複数室の住宅、外皮、日射、スケジュール | [`examples/vs_simheat_sample.py`](../examples/vs_simheat_sample.py) | 同じ層構造の Python |
| `vt.run_calc` の呼び方 | [`examples/run_calc_minimal.py`](../examples/run_calc_minimal.py) | 実行コード |

よく使う建物は `vs_simheat_sample.py` です。室名・気積・壁・窓・換気・発熱・空調の置き方は、このファイルの層を踏襲します。キーの意味が分からないときだけ [早見表](node_branch_schema.md) と [単位](units.md) を見ます。

## 2. 住宅は層に分けて書く

`vs_simheat_sample.py` は、壁のすべての層温度を JSON に展開していません。人間が直せる単位で分けています。同じ分け方にしてください。

```text
room_volume     室の気積
layers          壁・床・天井の層構成（一度だけ）
surface         方位・日射・窓種の部位カタログ
nodes           室と外部（気積から熱容量を付ける）
ventilation_branches  給気・排気・室間（vol は m³/s。m³/h なら / 3600）
surfaces        接続と面積だけ（key は「室A->室B||部位」、面積は式のまま）
heat_source / humidity_source / aircon
```

面積は数値に潰さず、一般部／熱橋や窓控除が見える式のまま置きます。例: `8.74 * 0.83`、`(10.92 - 4.59) * 0.17`。

層構成を変えるときは `layers`、同じ壁を別の室に付けるときは `surfaces` の面積行だけを足します。

## 3. やってはいけないこと

- 層ノード・熱枝まで手で展開した JSON を書く（それは builder の仕事です）
- 寸法・隣接・窓面積が与えられていないのに、それらしい住宅を作り上げる
- 風量を m³/h のまま `vol` に入れる
- `examples/run_calc_humidity_moisture_minimal.py` を雛形にする（給気のみで、`vol` の単位もこのリポジトリの約束と揃っていません）
- 未知のフィールドを JSON に足してごまかす。仮定は JSON の外に書く

足りない寸法・隣接・気象は、仮定した値と根拠を生成ログに残し、入力には入れた仮定だけを書きます。建物条件から JSON を自動で起こす処理は、この版にはありません。

## 4. 検査

API に届くときは、本計算の前に `POST /validate`（`unknown_keys=error`）と `POST /build` で直します。手順は [入力JSONの組み立て](ai_input_guide.md) の「本計算の前に検査する」です。気象ファイルや Colab 専用のセルは、検査用 JSON には入れません。

`vs_simheat_sample.py` は HASP 気象と pandas の時系列をその場で作ります。このファイルを `/validate` に貼ることはできません。検査するなら、短い期間に切った `input_data` を JSON 互換にして送ります。

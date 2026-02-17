# docs index

`vtsimnx` のドキュメント一覧です。  
用途ごとに、次の順で読むと把握しやすくなります。

## 1. 日射計算（太陽位置・直散分離・遮蔽）

- `solar_usage.md`
  - `solar_gain_by_angles`
  - `solar_gain_by_angles_with_shade`
  - 座標系（AZs）と入力パターン

## 2. 風圧・夜間放射・快適性

- `archenv_comfort_nocturnal_wind_usage.md`
  - `make_wind`
  - `nocturnal_gain_by_angles`
  - `calc_PMV` / `calc_PPD`
  - `calc_fungal_index`

## 3. ノード・枝スキーマ

- `node_branch_schema.md`
  - 計算入力のノード/枝データ構造

## 補足

- ルートの `README.md` はセットアップと実行手順の入口です。
- API単位の実装詳細は `vtsimnx/archenv/*.py` を参照してください。


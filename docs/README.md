# docs index

`vtsimnx` のドキュメント一覧です。  
用途ごとに、次の順で読むと把握しやすくなります。

## 0. 先に読む（建築環境工学の基礎）

- `building_environment_engineering_basics.md`
  - 熱収支の全体像
  - 日射（GHI/DNI/DHI）と方位・傾斜
  - 換気・風圧・湿気・夜間放射・地盤
  - PMV/PPD の読み方

## 1. 日射計算（太陽位置・直散分離・遮蔽）

- `solar_usage.md`
  - `solar_gain_by_angles`
  - `solar_gain_by_angles_with_shade`
  - 座標系（AZs）と入力パターン

## 2. 風圧・夜間放射・地盤温度・快適性

- `archenv_comfort_nocturnal_wind_usage.md`
  - `make_wind`
  - `nocturnal_gain_by_angles`
  - `ground_temperature_by_depth`
  - `calc_PMV` / `calc_PPD`
  - `calc_fungal_index`

## 3. ノード・枝スキーマ

- `node_branch_schema.md`
  - 計算入力のノード/枝データ構造

## 補足

- ルートの `README.md` はセットアップと実行手順の入口です。
- API単位の実装詳細は `vtsimnx/archenv/*.py` を参照してください。
- 物理モデルの背景理解が必要な場合は、まず `building_environment_engineering_basics.md` を参照してください。


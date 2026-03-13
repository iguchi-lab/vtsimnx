# examples

このディレクトリは、`vtsimnx` 利用者向けのサンプル Python コード置き場です。

## 収録サンプル

- `vs_simheat_r15.py`
  - SimHeat 比較ケースの入力作成〜`vt.run_calc` 実行〜結果比較までを含む大規模サンプルです。
- `3639999.has`
  - 上記サンプルで利用する気象データ（HASP）です。

## 置き方の目安

- 利用者がそのまま実行できる例を置く
- 1ファイル1テーマ（例: 最小 `run_calc`、`surfaces` 利用、スケジュール利用）
- ファイル名は内容が分かる名前にする
  - 例: `run_calc_minimal.py`, `run_calc_with_surfaces.py`

## 実行時の前提

- `vtsimnx` をインストール済みであること
- `vt.run_calc(...)` を使う例は、別途 API サーバー（`engine/`）が起動していること

## `vs_simheat_r15.py` 利用時の注意

- このファイルは Colab 由来のコードを含むため、`!pip ...` や `google.colab` 依存部分は、
  ローカル環境で実行する場合に調整が必要です。
- 気象ファイルパスは、リポジトリ内の `examples/3639999.has` を使うように置き換えると再現しやすくなります。

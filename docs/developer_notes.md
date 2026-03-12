# Developer Notes

このページは運用者・開発者向けの補足メモです。利用者向け情報は `README.md` と `docs/api_reference.md` を優先してください。

## リポジトリ対応

- 本ディレクトリは `vtsimnx-api` リポジトリ
- Python コアライブラリ側は別リポジトリ（`vtsimnx/core`）

## 運用メモ

- 常駐起動やログ確認は `RUN_FASTAPI.md` を参照
- solver 実行は `build/vtsimnx_solver` を利用
- artifact は `work/` 配下に生成される

## ドキュメント更新の優先順位

1. `README.md`（初見ユーザー向け）
2. `docs/api_reference.md`（API契約）
3. 詳細仕様（`docs/*.md`）
4. 本ファイル（内部メモ）

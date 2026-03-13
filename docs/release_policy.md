# release policy

このドキュメントは、`vtsimnx` のリリースとバージョニングの運用ルールです。  
利用者が「どのドキュメント/サンプルがどの版に対応するか」を迷わない状態を維持することを目的にします。

## 基本方針

- セマンティックバージョニング（`MAJOR.MINOR.PATCH`）を採用します。
- リリース時は、少なくとも次を同期します。
  - `pyproject.toml` の `project.version`
  - `engine/app/main.py` の FastAPI version
  - Git tag（例: `v1.0.0`）
  - GitHub Release ノート

## 互換性の目安

- **MAJOR**: 破壊的変更（API契約・入力仕様の非互換）
- **MINOR**: 後方互換の機能追加
- **PATCH**: バグ修正・ドキュメント修正

## ドキュメントとサンプルの対応

- ルート `README.md` は常に最新安定版の導線を示す
- `docs/` は最新版を基準に保守する
- `examples/` は最新 API で動作確認済みの状態を維持する
- 破壊的変更時は、該当ドキュメントに移行注意を明記する

## リリース手順（最小）

1. バージョン文字列を更新
2. ビルド/主要動作を確認（wheel build, 最小 run）
3. 変更を commit
4. タグ作成（`vX.Y.Z`）
5. `main` とタグを push
6. GitHub Release を作成
7. `README.md` の latest release リンクを必要に応じて更新

## チェックリスト

- [ ] `pyproject.toml` version 更新
- [ ] `engine/app/main.py` version 更新
- [ ] `python -m build --wheel` 成功
- [ ] `examples/run_calc_minimal.py` など最小導線の確認
- [ ] tag / release 作成
- [ ] release note に主変更点・注意点を記載

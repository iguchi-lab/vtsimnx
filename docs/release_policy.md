# release policy

このドキュメントは、`vtsimnx` のリリースとバージョニングの運用ルールです。
利用者が「どのドキュメント/サンプルがどの版に対応するか」を迷わない状態を維持することを目的にします。

## 基本方針

- セマンティックバージョニング（`MAJOR.MINOR.PATCH`）を採用します。
- **バージョンの正本は `pyproject.toml` の `project.version` のみ**です。
  - Python: `vtsimnx.get_version()` / `vtsimnx.__version__`
  - FastAPI / OpenAPI / `GET /version` の `api_version`: `engine/app/versioning.py` 経由で同じ値を参照
  - README の「最新リリース」リンクと Git tag（`vX.Y.Z`）はリリース時に正本へ合わせる
- API サーバー実装の正本は monorepo 内 `engine/` とする（別repo運用を前提にしない）

手動で FastAPI の `version=` 文字列を別途同期する必要はありません。

## 互換性の目安

- **MAJOR**: 破壊的変更（API契約・入力仕様の非互換）、deprecated API の削除
- **MINOR**: 後方互換の機能追加、experimental API の変更は許容
- **PATCH**: バグ修正・ドキュメント修正

公開 API の安定性区分は [`public_api.md`](public_api.md) を参照してください。

## ドキュメントとサンプルの対応

- ルート `README.md` は常に最新安定版の導線を示す
- `docs/` の現行ガイドは main の実装を基準に保守し、検証時の commit を記録する。リリース済みタグの仕様と一致するかはタグごとに確認する
- 技術文書の v3.2 は文書の版番号であり、パッケージ v1.7.2 とは別である。改訂時は DOCX と PDF を同期し、参照実装と改訂日を更新する
- v2.8、v3.1 等の旧技術文書は履歴資料として保持し、現行仕様の参照先としない
- `examples/` は最新 API で動作確認済みの状態を維持する
- 破壊的変更時は、該当ドキュメントに移行注意を明記する

## リリース手順（最小）

1. `pyproject.toml` の `project.version` だけを更新する
2. ビルド/主要動作を確認（`python -m build --wheel`, 最小 run）
3. 変更を commit
4. タグ作成（`vX.Y.Z`、正本と同じ番号）
5. `main` とタグを push
6. GitHub Release を作成
7. `README.md` の latest release リンクを必要に応じて更新

## チェックリスト

- [ ] `pyproject.toml` の `project.version` 更新（これだけが版の正本）
- [ ] `python -c "from vtsimnx import get_version; print(get_version())"` が正本と一致
- [ ] `GET /version` の `api_version` が正本と一致（engine 起動時）
- [ ] `python -m build --wheel` 成功
- [ ] `examples/run_calc_minimal.py` など最小導線の確認
- [ ] tag / release 作成（`v` + 正本）
- [ ] README の latest release リンク更新
- [ ] release note に主変更点・注意点・deprecated 削除予定を記載

# 旧 `vtsimnx-api` 廃止チェックリスト

モノレポ移行後に旧 `vtsimnx-api` リポジトリを安全に廃止するための手順です。

## 事前条件

- モノレポ `vtsimnx` 側の `api/` が正本として運用されている
- 主要ドキュメント（README/API仕様/運用手順）がモノレポ側で完結している

## 実施順

1. 旧repo README に移行告知と canonical URL を記載
2. 旧repoで Issue/PR 受付停止方針を明記
3. 社内外の参照リンク（Wiki/CI/運用手順）をモノレポURLへ更新
4. 2〜4週間の移行期間を設定し、利用者に告知
5. 移行期間後、旧repoを archive
6. 削除する場合は archive 後に最終確認して実施

## 最終確認項目

- 旧URLを参照する自動化（CI/CD, scripts, submodules）が残っていない
- 主要利用者がモノレポ導線でセットアップ可能
- 監視・運用手順が `api/RUN_FASTAPI.md` に統一されている

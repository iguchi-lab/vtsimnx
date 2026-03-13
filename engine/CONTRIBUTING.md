# Contributing

コントリビュートありがとうございます。最小限の開発フローをまとめます。

## 開発環境

- Python API: `app/`
- builder: `app/builder/`
- C++ solver: `solver/`

## テスト

- Python:

```bash
pytest
```

- C++ solver（通常）:

```bash
cmake -S solver -B build-solver -DCMAKE_BUILD_TYPE=Release
cmake --build build-solver
ctest --test-dir build-solver
```

- C++ solver（低負荷）:

```bash
cmake -S solver -B build-solver -DCMAKE_BUILD_TYPE=Release
cmake --build build-solver -j1
ctest --test-dir build-solver -j1 --output-on-failure
```

## C++ コンパイル方針（重要）

- 本番相当の実行確認は `Release` を基本にする
- SSH切断回避やCPU飽和回避が必要な環境では、`-j1`（必要なら `-j2`）を基本にする
- 高並列ビルドは、ユーザーが明示した場合のみ使う
- ビルド手順は `-S` / `-B` を明示して再現可能にする

推奨コマンド（再掲）:

```bash
cmake -S solver -B build-solver -DCMAKE_BUILD_TYPE=Release
cmake --build build-solver -j1
ctest --test-dir build-solver -j1 --output-on-failure
```

## 変更時の目安

- APIのI/Oを変えたら `docs/api_reference.md` を更新する
- builder 入力仕様を変えたら `docs/builder_json.md` を更新する
- 挙動を変えたら Python/C++ いずれかのテストを追加する

## Git運用

基本的な同期フロー:

```bash
git status -sb
git add <paths>
git commit -m "<message>"
git push
```

詳細なリポジトリ運用メモは `docs/developer_notes.md` を参照してください。

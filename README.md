# vtsimnx

[![CI](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml/badge.svg)](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml)

建築環境工学（熱・換気・湿気）を対象とした、研究/開発向けシミュレーション基盤です。  
Python クライアントで入力を構築し、HTTP engine で計算を実行する構成を提供します。

このリポジトリは **monorepo** です。  
`vtsimnx/`（Pythonライブラリ）と `engine/`（FastAPI+solver）を同じリポジトリで保守します。
APIサーバー実装の正本は `engine/` です。

最新リリース: [`v1.0.3`](https://github.com/iguchi-lab/vtsimnx/releases/tag/v1.0.3)

バージョン整合ポリシー（`pyproject.toml` / FastAPI version / tag）: `docs/release_policy.md`

## 何ができるか

- `vt.run_calc(...)` を使った回路網計算の実行（engine `/run` 呼び出し）
- `surfaces` / `aircon` / `heat_source` を含む入力JSONの組み立て
- artifact（結果ファイル、ログ、スキーマ）取得と比較評価
- 日射/夜間放射/地盤温度/スケジュール等の補助計算

## 構成図

```text
Python Client (vtsimnx/)
  -> input_data (dict/json)
  -> POST /run
HTTP Engine (engine/)
  -> builder (入力正規化)
  -> C++ solver
  -> artifacts / result files
Docs (docs/, engine/docs/)
Examples (examples/)
```

## 3つの開始方法

### 1) クライアント API だけ読む

- 入口: `docs/README.md`
- 最短: `docs/builder_input_quickstart.md` -> `docs/node_branch_schema.md`

### 2) ローカルで engine を起動する

```bash
cd engine
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -sS http://127.0.0.1:8000/ping
```

詳細: `engine/RUN_FASTAPI.md`

### 3) サンプルを動かす

```bash
python examples/run_calc_minimal.py
```

大規模ケース: `examples/vs_simheat_sample.py`  
サンプル一覧: `examples/README.md`

## ドキュメント導線

- 利用者向け入口: `docs/README.md`
- engine 実装仕様入口: `engine/docs/README.md`
- API契約: `engine/docs/api_reference.md`
- builder 入力仕様（正本）: `engine/docs/builder_json.md`
- 検証戦略: `docs/validation_strategy.md`
- リリース運用: `docs/release_policy.md`

## CI（公開チェック）

`main` push / PR ごとに GitHub Actions で以下を自動実行します。

- `ruff check vtsimnx/materials`
- `mypy --ignore-missing-imports --follow-imports=skip vtsimnx/materials/__init__.py`
- `python -m pytest -q vtsimnx/tests/test_utils_io.py vtsimnx/tests/test_run_calc_lazy.py`

## 検証と保証範囲

本プロジェクトは研究用途のため、検証方針と既知の限界を公開しています。  
何を保証し、何を未保証としているかは `docs/validation_strategy.md` を参照してください。

## リポジトリ構成

- `vtsimnx/`: Python client ライブラリ
- `engine/`: FastAPI + builder + C++ solver
- `examples/`: 実行サンプル
- `docs/`: 利用者向けドキュメント
- `engine/docs/`: engine 実装仕様ドキュメント

## License / Disclaimer

- ライセンス: MIT (`LICENSE`)
- 本ソフトウェアは研究・開発目的で提供され、結果の正確性・完全性・特定目的適合性は保証されません。
- 運用利用前の入力条件・仮定・出力結果の妥当性確認は利用者の責任で実施してください。

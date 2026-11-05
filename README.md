# vtsimnx

[![CI](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml/badge.svg)](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml)

建築環境工学（熱・換気・湿気）を対象とした、研究/開発向けシミュレーション基盤です。  
Python クライアントで入力を構築し、HTTP engine で計算を実行する構成を提供します。

このリポジトリは **monorepo** です。  
`vtsimnx/`（Pythonライブラリ）と `engine/`（FastAPI+solver）を同じリポジトリで保守します。
APIサーバー実装の正本は `engine/` です。

最新リリース: [GitHub Releases](https://github.com/iguchi-lab/vtsimnx/releases/latest)

mainのv1.8.1修正内容と移行上の注意：[領域別点検の修正記録](docs/releases/v1.8.1.md)。

バージョン正本は `pyproject.toml`（API / `get_version()` / tag が参照）。運用: `docs/release_policy.md`  
公開 API の安定性: `docs/public_api.md`　単位系: `docs/units.md`

## 何ができるか

- `vt.run_calc(...)` を使った回路網計算の実行（engine `/runs` ポーリング、互換で `/run` も可）
- `surfaces` / `aircon` / `heat_source` を含む入力JSONの組み立て
- **エアコン未設置空間の温度制御**（`set` と `in`/`out` を分離。例: 階間設置で LDK を制御）
- artifact（結果ファイル、ログ、スキーマ）取得と比較評価
- 日射/夜間放射/地盤温度/スケジュール等の補助計算

## 構成図

```text
Python Client (vtsimnx/)
  -> input_data (dict/json)
  -> POST /validate, /build（任意）
  -> POST /runs
HTTP Engine (engine/)
  -> builder（展開・検証）
  -> C++ solver
  -> artifacts
```

## 3つの開始方法

### 1) 入力の書き方を読む

- 文書の層と読み順: [`docs/README.md`](docs/README.md)
- 入力の意味: [`docs/ai_input_guide.md`](docs/ai_input_guide.md)
- AI にモデル作成を頼むとき: [`docs/ai_authoring.md`](docs/ai_authoring.md)
- 実行コード: [`docs/builder_input_quickstart.md`](docs/builder_input_quickstart.md)

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

## ドキュメント

文書は4層です。上から読んでください。詳細な案内は [`docs/README.md`](docs/README.md) です。

| 層 | 場所 | 用途 |
|---|---|---|
| 使い方 | [`docs/`](docs/README.md) | 入力の意味、単位、機能ガイド |
| 実装契約 | [`engine/docs/`](engine/docs/README.md) | フィールド正本、HTTP、計算ループ |
| 規範仕様 | [`docs/specification/`](docs/specification/README.md) | 章立ての仕様本文 |
| 解説・記録 | [技術解説書 v3.3](docs/vtsimnx_calculation_model_and_implementation_flow_ja_v3.3.pdf) | モデル解説と点検メモ |

入力 JSON の入口: [`docs/ai_input_guide.md`](docs/ai_input_guide.md)  
AI への指令: [`docs/ai_authoring.md`](docs/ai_authoring.md)  
単位: [`docs/units.md`](docs/units.md)　公開 API: [`docs/public_api.md`](docs/public_api.md)

## CI（公開チェック）

`main` push / PR ごとに GitHub Actions で以下を自動実行します。

| ジョブ | 内容 |
|---|---|
| `lint` | `ruff` + `mypy`（Python 3.11） |
| `python-client` | `pytest vtsimnx/tests`（Python **3.10 / 3.11 / 3.13**） |
| `engine-python` | C++ solver ビルド + `pytest engine/tests_py`（physics/perf 除外） |
| `physics-regression` | physics marker のベースライン回帰（Python 3.11） |
| `perf-history` | 性能ベンチ（warn-only, Python 3.11） |
| `cpp-solver` | CMake ビルド + CTest |
| `package` | wheel ビルド → クリーン環境へ install → import 確認（Python 3.11） |
| `example` | uvicorn 起動 + `examples/run_calc_minimal.py`（Python 3.11） |

対応 Python: **3.10 以上**（`pyproject.toml` の `requires-python`）。lint / package / example は主要版 3.11 固定。

ワークフロー定義: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
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

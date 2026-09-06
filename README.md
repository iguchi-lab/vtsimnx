# vtsimnx

[![CI](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml/badge.svg)](https://github.com/iguchi-lab/vtsimnx/actions/workflows/ci.yml)

建築環境工学（熱・換気・湿気）を対象とした、研究/開発向けシミュレーション基盤です。  
Python クライアントで入力を構築し、HTTP engine で計算を実行する構成を提供します。

このリポジトリは **monorepo** です。  
`vtsimnx/`（Pythonライブラリ）と `engine/`（FastAPI+solver）を同じリポジトリで保守します。
APIサーバー実装の正本は `engine/` です。

最新リリース: [`v1.7.2`](https://github.com/iguchi-lab/vtsimnx/releases/tag/v1.7.2)

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

| 対象 | 入口 |
|---|---|
| 外部研究者向け計算モデル・実装フロー解説 | [`技術解説書 v3.1（PDF）`](docs/vtsimnx_calculation_model_and_implementation_flow_ja_v3.1.pdf) / [`Word版`](docs/vtsimnx_calculation_model_and_implementation_flow_ja_v3.1.docx) |
| 利用者（入力の書き方） | [`docs/README.md`](docs/README.md) |
| engine 実装仕様 | [`engine/docs/README.md`](engine/docs/README.md) |
| 単位の正本 | [`docs/units.md`](docs/units.md) |
| 公開 API 安定性 | [`docs/public_api.md`](docs/public_api.md) |
| 検証方針 | [`docs/validation_strategy.md`](docs/validation_strategy.md) |
| リリース運用 | [`docs/release_policy.md`](docs/release_policy.md) |

入力 JSON の厳密仕様（正本）: [`engine/docs/builder_json.md`](engine/docs/builder_json.md)

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

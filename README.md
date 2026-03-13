# vtsimnx
熱・換気回路網計算に基づくシミュレーション基盤です。  
このリポジトリは `vtsimnx` Python ライブラリ（core）と FastAPI 計算サーバー（`engine/`）を同居させています。

## 利用目的別クイックリンク

- ライブラリ利用（外部実行クライアント）: `vtsimnx/`, `docs/README.md`
- 入力JSONの書き方（`vt.run_calc` 利用者向け）: `docs/builder_input_quickstart.md`, `docs/node_branch_schema.md`
- APIサーバー運用（計算実行）: `engine/README.md`, `engine/RUN_FASTAPI.md`
- API仕様・入力契約: `engine/docs/api_reference.md`, `engine/docs/builder_json.md`
- 開発者向け: `engine/CONTRIBUTING.md`

## クイックスタート（core ライブラリ）

1) 仮想環境の作成と依存導入

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

2) テスト実行

```bash
python -m pytest
```

3) `run_calc` 疎通確認（API稼働前提）

```bash
python -m vtsimnx.tools.run_calc_smoke --base-url http://127.0.0.1:8000
```

## API連携の前提

- `vt.run_calc(...)` は `engine/` の `/run` エンドポイントを呼び出します。
- API URL は引数で直接渡すか、`VTSIMNX_API_URL` を利用します。
- APIの起動・常駐運用は `engine/RUN_FASTAPI.md` を参照してください。

## ドキュメント構成

- core側の理論・ユーティリティ: `docs/README.md`
- 利用者向けの入力組み立て導線: `docs/builder_input_quickstart.md`
- API側の実装/契約ドキュメント: `engine/docs/README.md`
- ノード/枝の利用者向け整理: `docs/node_branch_schema.md`
- builder厳密仕様（正本）: `engine/docs/builder_json.md`

## リポジトリ構成

- `vtsimnx/`: 外部実行者が利用する Python ライブラリ群
- `docs/`: core側ドキュメント（理論・使用例）
- `engine/`: FastAPI + builder + C++ solver（計算サーバー）

## License / Disclaimer

- ライセンス: MIT (`LICENSE`)
- 本ソフトウェアは研究・開発目的で提供され、結果の正確性・完全性・特定目的適合性は保証されません。
- 運用利用前の入力条件・仮定・出力結果の妥当性確認は利用者の責任で実施してください。

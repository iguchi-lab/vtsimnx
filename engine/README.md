# VTSimNX Engine Service

VTSimNX のシミュレーションを HTTP で実行する FastAPI サービスです。  
JSON を受け取ると builder で正規化し、C++ solver を実行して結果と artifact を返します。

- リポジトリ全体: [`../README.md`](../README.md)
- 使い方（入力の意味）: [`../docs/README.md`](../docs/README.md)
- 実装契約の索引: [`docs/README.md`](docs/README.md)

## エンドポイント（概要）

- `GET /ping` — ヘルスチェック
- `POST /validate` — 入力検査（ソルバなし）
- `POST /build` — builder 展開（ソルバなし）
- `POST /runs` — 非同期実行（推奨）
- `POST /run` — 同期実行（互換）
- `GET /artifacts/...` — 成果物取得

詳細: [`docs/api_reference.md`](docs/api_reference.md)

## クイックスタート

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -sS http://127.0.0.1:8000/ping
```

```bash
curl -sS -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "simulation": {
        "index": {
          "start": "2026-01-01T00:00:00",
          "end": "2026-01-01T01:00:00",
          "timestep": 3600,
          "length": 2
        }
      },
      "nodes": [{"key": "outside", "t": 5.0}, {"key": "room", "calc_t": true, "v": 30.0}],
      "ventilation_branches": [],
      "thermal_branches": []
    }
  }'
```

運用の詳細は [`RUN_FASTAPI.md`](RUN_FASTAPI.md) を参照してください。

## ドキュメント

使い方は [`../docs/README.md`](../docs/README.md) です。ここにあるのは実装契約です。

| 文書 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | 実装契約の索引 |
| [`docs/api_reference.md`](docs/api_reference.md) | HTTP API 契約 |
| [`docs/builder_json.md`](docs/builder_json.md) | builder 入力正本 |
| [`docs/simulation_overview.md`](docs/simulation_overview.md) | 計算フロー |
| [`docs/aircon_control_overview.md`](docs/aircon_control_overview.md) | 空調制御 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 開発参加・テスト |

## License / Disclaimer

- MIT: [`../LICENSE`](../LICENSE)
- 研究・開発目的。結果の正確性は保証しません。運用前の妥当性確認は利用者責任です。

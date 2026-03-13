# API Reference

このドキュメントは `app/main.py` の FastAPI エンドポイント仕様をまとめたものです。

## Base URL

- 例: `http://127.0.0.1:8000`

## 認証

- 現在は認証なし

## エンドポイント一覧

| Method | Path | 用途 |
|---|---|---|
| GET | `/ping` | ヘルスチェック |
| POST | `/run` | シミュレーション実行 |
| GET | `/artifacts/{artifact_dir}/manifest` | 実行結果メタ情報取得 |
| GET | `/artifacts/{artifact_dir}/files` | ダウンロード可能キー一覧取得 |
| GET | `/artifacts/{artifact_dir}/download/{key}` | artifact ファイル実体ダウンロード |

---

## GET /ping

### Response 200

```json
{"status":"ok"}
```

---

## POST /run

builder 入力 (`config`) を受け取り、solver 実行結果を返します。

### Request Body

```json
{
  "config": {},
  "debug": false,
  "debug_verbosity": 2,
  "add_surface": null,
  "add_aircon": null,
  "add_capacity": null,
  "add_moisture_capacity": null,
  "add_surface_solar": null,
  "add_surface_nocturnal": null,
  "add_surface_radiation": null,
  "add_surface_radiation_exclude_glass": null
}
```

- `config` (required): builder 入力 JSON
- `debug` (optional, default `false`): ログ冗長度制御をデバッグ寄りにする
- `debug_verbosity` (optional, default `2`): `debug=true` 時の最小 verbosity
- `add_*` 系 (optional): builder の各展開処理を API から上書き制御

### Response 200

```json
{
  "result": {
    "artifact_dir": "run_20260312_abcdef12",
    "result_files": {
      "schema": "schema.json"
    },
    "log_file": "solver.log"
  },
  "warnings": [],
  "warning_details": []
}
```

### Error Response

#### 400 Bad Request（入力不正）

```json
{
  "detail": {
    "code": "invalid_config",
    "message": "..."
  }
}
```

代表的な `detail.code`:

- `invalid_config`
- `invalid_config_missing_field`

#### 500 Internal Server Error（実行時エラー）

```json
{
  "detail": {
    "code": "internal_error",
    "message": "...",
    "run_id": "..."
  }
}
```

代表的な `detail.code`:

- `internal_error`
- `solver_binary_not_found`
- `solver_execution_failed`

---

## GET /artifacts/{artifact_dir}/manifest

artifact ディレクトリ配下の `manifest.json` を返します。

### Response 200（例）

```json
{
  "created_at": "2026-03-12T00:00:00+00:00",
  "output": {
    "artifact_dir": "run_20260312_abcdef12",
    "result_files": {
      "schema": "schema.json"
    },
    "log_file": "solver.log",
    "builder_log_file": "builder.log"
  },
  "result_files": {
    "schema": "schema.json"
  },
  "files": {
    "schema": "schema.json",
    "log": "solver.log",
    "builder_log": "builder.log",
    "manifest": "manifest.json"
  }
}
```

---

## GET /artifacts/{artifact_dir}/files

`download/{key}` に渡せるキー一覧を返します。

### Response 200（例）

```json
{
  "artifact_dir": "run_20260312_abcdef12",
  "keys": ["schema", "log", "builder_log", "manifest"]
}
```

---

## GET /artifacts/{artifact_dir}/download/{key}

キーで指定したファイルを返します。  
`key` は `/artifacts/{artifact_dir}/files` で取得したもののみ利用してください。

### 例

```bash
curl -L -o schema.json http://127.0.0.1:8000/artifacts/<artifact_dir>/download/schema
curl -L -o solver.log  http://127.0.0.1:8000/artifacts/<artifact_dir>/download/log
```

---

## 補足

- `Content-Encoding: gzip` のリクエストボディを受け付けます。
- `artifact_dir` および配布ファイルはパストラバーサル対策済みです。
- OpenAPI は起動後に `/docs`（Swagger UI）と `/openapi.json` でも参照できます。

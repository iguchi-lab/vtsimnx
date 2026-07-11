# API Reference

このドキュメントは `app/main.py` の FastAPI エンドポイント仕様をまとめたものです。

## Base URL

- 例: `http://127.0.0.1:8000`

## 認証

- `X-API-Key` ヘッダ（環境変数 `VTSIMNX_API_KEYS` で有効化。未設定時は認証なし）

## エンドポイント一覧

| Method | Path | 用途 |
|---|---|---|
| GET | `/ping` | ヘルスチェック |
| POST | `/runs` | 非同期ジョブ投入（推奨） |
| GET | `/runs/{run_id}` | ジョブ状態・進捗 |
| GET | `/runs/{run_id}/result` | 完了時の結果取得 |
| DELETE | `/runs/{run_id}` | ジョブキャンセル |
| POST | `/run` | 同期シミュレーション（互換・デバッグ用） |
| GET | `/artifacts/{artifact_dir}/manifest` | 実行結果メタ情報取得 |
| GET | `/artifacts/{artifact_dir}/files` | ダウンロード可能キー一覧取得 |
| GET | `/artifacts/{artifact_dir}/download/{key}` | artifact ファイル実体ダウンロード |

ジョブ表はプロセス内メモリ上に保持します。**uvicorn は `workers=1` を推奨**します（マルチワーカーではジョブ状態を共有しません）。並列度は `VTSIMNX_MAX_WORKERS`（既定 1）で制御します。

---

## GET /ping

### Response 200

```json
{"status":"ok"}
```

---

## POST /runs

非同期ジョブを投入し、即時に `run_id` を返します。同一入力（canonical JSON の sha256）が `queued` / `running` の既存ジョブと一致する場合は、その `run_id` を **200** で返します。

### Request Body

`POST /run` と同じ（`config` / `debug` / `add_*`）。

### Response 202（新規） / 200（重複）

```json
{
  "run_id": "a1b2c3...",
  "status": "queued",
  "input_hash": "sha256hex..."
}
```

## GET /runs/{run_id}

### Response 200

```json
{
  "run_id": "a1b2c3...",
  "status": "running",
  "input_hash": "...",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "progress": {"stage": "solver", "message": "running solver"},
  "artifact_dir": null,
  "error": null
}
```

`status`: `queued` | `running` | `succeeded` | `failed` | `cancelled`

## GET /runs/{run_id}/result

完了時のみ `POST /run` と同じ `SimulationResponse` を返します。未完了は **409**。

## DELETE /runs/{run_id}

`queued` は即キャンセル。`running` はキャンセルフラグを立て、可能なら solver 子プロセスを terminate します。

---

## POST /run

builder 入力 (`config`) を受け取り、solver 実行結果を**同期**で返します（互換 API。長時間計算は `POST /runs` を推奨）。

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
- 成果物取得は `/artifacts/...` に一本化しています。`/work` の静的公開は行いません。
- OpenAPI は起動後に `/docs`（Swagger UI）と `/openapi.json` でも参照できます。

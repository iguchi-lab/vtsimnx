# FastAPI 起動メモ（`app/main.py`）

## 最短コマンド（venv あり）

```bash
cd /home/ubuntu/vtsimnx-api
/home/ubuntu/vtsimnx-api/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

起動確認:

```bash
curl -sS http://127.0.0.1:8000/ping
```

## ログについて

C++ソルバは `output.json` に `artifact_dir` と `log_file` を返します。ログ本体は **artifact 配下の `solver.log`** として出力されるため、ログは **ダウンロードAPIで取得**してください。

## 巨大な計算結果（artifact）をAPIで取得する

`/run` のレスポンスに含まれる `artifact_dir` を使って、API経由でファイルをダウンロードできます（OSのファイル共有は不要）。

利用手順:

1) 実行して `artifact_dir` を得る

2) 取得可能なキー一覧を確認

```bash
curl -sS http://127.0.0.1:8000/artifacts/<artifact_dir>/files
```

3) 取得（例: schema / vent_pressure / log）

```bash
curl -L -o schema.json http://127.0.0.1:8000/artifacts/<artifact_dir>/download/schema
curl -L -o vent.pressure.f32.bin http://127.0.0.1:8000/artifacts/<artifact_dir>/download/vent_pressure
curl -L -o solver.log http://127.0.0.1:8000/artifacts/<artifact_dir>/download/log
```

## ログ冗長度（verbosity）のAPI側制御（debug時だけ増やす）

`/run` は API側で `simulation.log.verbosity` を強制します。

- `debug` 未指定/`false` の場合: **常に `verbosity=1`**（入力JSONに指定があっても上書き）
- `debug=true` の場合: **`verbosity` を `debug_verbosity` まで引き上げ**（既に高い場合は維持）

例（デバッグON）:

```bash
curl -sS -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"config": {...}, "debug": true, "debug_verbosity": 2}'
```

## bash で単発実行（APIの /run と同じ経路でデバッグしたい）

例（quiet: verbosity=0 / silent）:

```bash
python3 -m app.main work_test/vs_simheat_8760.json --quiet
```

例（debug: verbosityを引き上げ）:

```bash
python3 -m app.main work_test/vs_simheat_8760.json --debug --debug-verbosity 2
```

例（明示指定）:

```bash
python3 -m app.main work_test/vs_simheat_8760.json --verbosity 1
```

## スクリプトで起動（推奨）

### 常駐で起動（ON）/ 停止（OFF）/ 状態確認

ON:

```bash
cd /home/ubuntu/vtsimnx-api
./scripts/start_api.sh
```

ログ（常駐起動時は `.uvicorn.log` に出ます）:

```bash
./scripts/log_api.sh
```

OFF:

```bash
./scripts/stop_api.sh
```

状態:

```bash
./scripts/status_api.sh
```

### フォアグラウンドで起動（ログを見たいとき）

```bash
cd /home/ubuntu/vtsimnx-api
./scripts/run_api.sh
```

ポート変更例:

```bash
PORT=8080 ./scripts/run_api.sh
```

起動確認:

```bash
./scripts/ping_api.sh
```



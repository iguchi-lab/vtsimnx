# FastAPI 起動と運用

このページは「どう起動するか」「運用時に何を見るか」の実務メモです。  
API 仕様（エンドポイント、入力、レスポンス、エラー）は `docs/api_reference.md` を参照してください。

## 1. 最短起動

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

確認:

```bash
curl -sS http://127.0.0.1:8000/ping
# {"status":"ok"}
```

開発時のみ:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 2. 常駐起動（ログ付き）

```bash
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> .uvicorn.log 2>&1 &
echo $! > .uvicorn.pid
```

ログ確認:

```bash
tail -f .uvicorn.log
```

停止:

```bash
kill "$(cat .uvicorn.pid)"
```

venv 例:

```bash
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> .uvicorn.log 2>&1 &
echo $! > .uvicorn.pid
```

## 3. API 利用の最小フロー

1) `/run` を呼び出して `artifact_dir` を得る  
2) `/artifacts/{artifact_dir}/files` でキー一覧を得る  
3) `/artifacts/{artifact_dir}/download/{key}` で必要ファイルを取得する

例:

```bash
curl -sS -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"config": {"simulation": {"step": 1, "timestep": 3600}, "nodes": [], "ventilation_branches": [], "thermal_branches": []}}'
```

```bash
curl -sS http://127.0.0.1:8000/artifacts/<artifact_dir>/files
curl -L -o schema.json http://127.0.0.1:8000/artifacts/<artifact_dir>/download/schema
curl -L -o solver.log http://127.0.0.1:8000/artifacts/<artifact_dir>/download/log
```

## 4. ログと artifact

- solver ログ本体は artifact 配下の `solver.log`
- builder ログは一時出力後、artifact 配下の `builder.log` に取り込み
- `manifest.json` には `result_files` と互換用 `files` マップを保存

## 5. よくある運用トラブル

| 症状 | 原因候補 | 対策 |
|---|---|---|
| しばらくして落ちる | `--reload` 利用 | 常駐運用は `--reload` なし |
| `/run` が返らない | solver ハング | `VTSIMNX_SOLVER_TIMEOUT` を設定 |
| プロセスが突然消える | OOM killer | 計算サイズ見直し、メモリ増強、システムログ確認 |
| 単発エラーで止まる | 単一ワーカー | systemd / supervisord で自動再起動 |

タイムアウト例:

```bash
export VTSIMNX_SOLVER_TIMEOUT=3600
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. 落ちたときの確認順

1. `.uvicorn.log`（または標準出力のリダイレクト先）  
2. `dmesg -T` / `/var/log/syslog` の OOM 記録  
3. 直前 run の artifact 内 `solver.log` / `builder.log`  
4. systemd 管理時は `journalctl -u <service-name> -n 200 --no-pager`

## 7. CLI 単発実行（/run と同経路）

```bash
python3 -m app.main input.json --quiet
python3 -m app.main input.json --debug --debug-verbosity 2
python3 -m app.main input.json --verbosity 1
```

## 8. 参考

- API仕様: `docs/api_reference.md`
- builder入力仕様: `docs/builder_json.md`
- 全体計算フロー: `docs/simulation_overview.md`



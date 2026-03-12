# FastAPI 起動メモ（`app/main.py`）

## API を起動する

リポジトリの **api** ディレクトリで以下を実行する（常駐・ログ付き）。`--reload` は付けない。

```bash
cd /home/ubuntu/vtsimnx/api
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> .uvicorn.log 2>&1 &
echo $! > .uvicorn.pid
```

起動確認:

```bash
curl -sS http://127.0.0.1:8000/ping
# => {"status":"ok"}
```

ログの確認:

```bash
tail -f .uvicorn.log
```

停止する場合（PID で kill）:

```bash
kill $(cat .uvicorn.pid)
```

venv を使う場合の例:

```bash
cd /home/ubuntu/vtsimnx/api
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> .uvicorn.log 2>&1 &
echo $! > .uvicorn.pid
```

---

## プロセスがしばらくすると落ちる場合

よくある原因と対策:

| 原因 | 対策 |
|------|------|
| **`--reload`** | 開発用。ファイル監視で再起動したり、子プロセス落ちで親も落ちることがある。**常駐運用では `--reload` を外す**。 |
| **ソルバのハング** | `/run` で C++ ソルバが応答しないとワーカーがブロックする。環境変数 `VTSIMNX_SOLVER_TIMEOUT`（秒）を設定すると、その秒数で打ち切り可能（未設定は無制限）。 |
| **メモリ不足 (OOM)** | 巨大な config や連続した `/run` でメモリが増え、OS に kill される。`dmesg` や `/var/log/syslog` で OOM を確認。計算サイズを減らすか、メモリを増やす。 |
| **単一ワーカー** | 1プロセスなので、未処理の例外や子プロセス異常でプロセス全体が終了する。systemd / supervisord で自動再起動を入れるとよい。 |

常駐用の起動例（**--reload なし**）:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

ソルバのタイムアウトを 1 時間にしたい場合:

```bash
export VTSIMNX_SOLVER_TIMEOUT=3600
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 最短コマンド（開発用・--reload あり）

```bash
cd /home/ubuntu/vtsimnx/api
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

※ 上記は開発用（`--reload` あり）。常駐運用では「API を起動する」または「プロセスがしばらくすると落ちる場合」を参照。

## ログについて

- C++ソルバは `output.json` に `artifact_dir` と `log_file` を返します。ログ本体は **artifact 配下の `solver.log`** として出力されるため、ログは **ダウンロードAPIで取得**してください。
- builder のログは、リクエストごとに一時ファイルへ出力したあと **artifact 直下の `builder.log` に取り込まれ、一時ファイルは自動削除**されます（`work/logs` は一時置き場として使われますが、残らない設計です）。

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

## スクリプトで起動（scripts がある場合）

`scripts/start_api.sh` が用意されている環境では、`cd /home/ubuntu/vtsimnx/api && ./scripts/start_api.sh` で常駐起動、`./scripts/log_api.sh` でログ確認（`.uvicorn.log`）。**scripts が無い場合は**、本文の「API を起動する」のコマンドを使う。

---

## 落ちたときのログ確認

FastAPI/uvicorn が落ちた**理由**を調べるには、次の順で確認する。

### 1. Uvicorn の標準出力・ログファイル

- **起動方法で変わる**:
  - `uvicorn ... &` や `nohup uvicorn ...` で、リダイレクト先（例: `nohup ... > .uvicorn.log 2>&1`）を見る。
  - `./scripts/start_api.sh` を使っている場合は、そのスクリプトがログをどこに書いているか（例: `api/.uvicorn.log` や `scripts/.uvicorn.log`）を確認する。
- **ログが無い場合**: 次回から「ログを残して起動」の例（下記）で起動すると、落ちた直前のメッセージがファイルに残る。

**ログを残して起動する例**（api ディレクトリで）:

```bash
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> .uvicorn.log 2>&1 &
echo $! > .uvicorn.pid
# 確認
tail -n 200 .uvicorn.log
```

### 2. システムログ（OOM や kill）

- **OOM killer**: メモリ不足でプロセスが kill されるとカーネルが記録する。
  ```bash
  dmesg -T | grep -i oom
  # または
  grep -i "out of memory\|oom\|killed process" /var/log/syslog
  ```
- **systemd で動かしている場合**:
  ```bash
  journalctl -u サービス名 -n 200 --no-pager
  ```

### 3. 直前に実行した /run のログ

- 落ちる直前に `/run` を叩いていた場合、**そのリクエスト**の builder ログは `/tmp/vtsimnx.builder.<run_id>.log` に残っている（成功時は artifact にコピー後に削除されるが、プロセスが落ちた場合は残ることがある）。
- C++ ソルバのログは、artifact の `solver.log`。プロセス落ちで artifact が書き終わっていない場合は、`work/` 配下の `solver.log` や実行ディレクトリを確認する。

### 4. 今回ログが無かった場合

- 現在の環境では **api 配下に `.uvicorn.log` は無く**、uvicorn をどう起動したか（どのディレクトリで、リダイレクトありか）によってログの有無が変わる。
- **次回から**上記「ログを残して起動する例」で起動するか、`scripts/start_api.sh` で確実に `.uvicorn.log` に出すようにすると、次に落ちたときに `tail .uvicorn.log` で原因を追いやすくなる。

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
cd /home/ubuntu/vtsimnx/api   # ワークスペース統合時は vtsimnx/api
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



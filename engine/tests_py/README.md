### tests_py（Python用テスト）
このフォルダは **Python（FastAPI / builder / Pythonラッパ）用のテスト**です。

### 実行方法
リポジトリルートで以下を実行します。

```bash
pytest
```

`pytest.ini` で `testpaths = tests_py` を設定しているため、`pytest` はこのフォルダ配下のみ収集・実行します。

物理基準ケースのみ:

```bash
python -m pytest tests_py -m physics
```

性能履歴のみ:

```bash
python tests_py/perf/run_bench.py --output /tmp/perf_report.json
python -m pytest tests_py -m perf
```

非物理・非性能（builder / API 等）のみ:

```bash
python -m pytest tests_py -m "not physics and not perf"
```

### どこまでカバーしているか（現状）

- **builder**: parse（`||`, `&&`, `A->B->C`展開）/ surfaces展開 / aircon展開 / thermal_mass→capacity変換 / validation（未知キー削除、type推定、重複key処理、response_conductionの係数チェック）
- **API層**: `/run` のI/Oやgzip受理など（solverはモック）
- **solver_runner**: IOパス/後始末/エラーハンドリング（solverバイナリ有無でskipするテストあり）
- **物理回帰（`physics/`）**: pytest marker `physics`。公開 CI の `physics-regression` ジョブで実行
  - golden に加え、エネルギー/質量収支残差・非負制約・収束ログ・NaN/Inf 不在を明示検証
  - 許容値の根拠: `physics/tolerances.py`
  - カテゴリ: 単室熱収支、RC vs 応答、複数室換気、日射/夜間放射、湿気、汚染物質、空調 ON/OFF、硬い圧力網、極端パラメータ、時間刻み収束
- **性能履歴（`perf/`）**: pytest marker `perf`。公開 CI の `perf-history` ジョブで実行
  - 代表ケースの builder/solver 時間、最大メモリ、LU 再構築、空調再計算、artifact サイズを記録
  - baseline (`perf/baselines/representative.json`) 比で大幅退行を WARNING（厳密合否はしない）

### C++（solver）のテストについて
C++ 側のテストは `pytest` とは別系統（CMake/ctest）で実行します。  
詳細は `../docs/cpp_test_catalog.md` および公開 CI の `cpp-solver` ジョブを参照してください。

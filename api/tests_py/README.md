### tests_py（Python用テスト）
このフォルダは **Python（FastAPI / builder / Pythonラッパ）用のテスト**です。

### 実行方法
リポジトリルートで以下を実行します。

```bash
pytest
```

`pytest.ini` で `testpaths = tests_py` を設定しているため、`pytest` はこのフォルダ配下のみ収集・実行します。

### どこまでカバーしているか（現状）

- **builder**: parse（`||`, `&&`, `A->B->C`展開）/ surfaces展開 / aircon展開 / thermal_mass→capacity変換 / validation（未知キー削除、type推定、重複key処理、response_conductionの係数チェック）
- **API層**: `/run` のI/Oやgzip受理など（solverはモック）
- **solver_runner**: IOパス/後始末/エラーハンドリング（solverバイナリ有無でskipするテストあり）
- **物理回帰（追加）**: `test_solver_physics_regression.py` で
  - 2層壁RCケースの室温系列を golden 比較
  - RC法と応答係数法（等価U値ケース）の数値回帰比較

### まだ薄いところ（今後の拡充候補）

- **C++ solver本体のユニットテスト**は `tests_py` では扱っていません（別系統で管理推奨）。
- 物理回帰は最小ケース中心のため、実務ケース（複数室・換気連成・日射/夜間放射）の golden 追加余地があります。

### C++（solver）のテストについて
C++ 側のテストは `pytest` とは別系統（例: CMake/ctest）で実行する前提です。
将来的に C++ テストを追加する場合は `solver/` 配下に配置し、CI も別ジョブで回すのが分かりやすいです。



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

### まだ薄いところ（今後の拡充候補）

- **C++ solver本体のユニットテスト**は `tests_py` では扱っていません（別系統で管理推奨）。
- **物理量の妥当性（golden比較）**や「RC vs response」の数値的な回帰は、現状 `tests_py` だけでは手薄です。

### C++（solver）のテストについて
C++ 側のテストは `pytest` とは別系統（例: CMake/ctest）で実行する前提です。
将来的に C++ テストを追加する場合は `solver/` 配下に配置し、CI も別ジョブで回すのが分かりやすいです。



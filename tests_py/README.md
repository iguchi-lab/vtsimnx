### tests_py（Python用テスト）
このフォルダは **Python（FastAPI / builder / Pythonラッパ）用のテスト**です。

### 実行方法
リポジトリルートで以下を実行します。

```bash
pytest
```

`pytest.ini` で `testpaths = tests_py` を設定しているため、`pytest` はこのフォルダ配下のみ収集・実行します。

### C++（solver）のテストについて
C++ 側のテストは `pytest` とは別系統（例: CMake/ctest）で実行する前提です。
将来的に C++ テストを追加する場合は `solver/` 配下に配置し、CI も別ジョブで回すのが分かりやすいです。



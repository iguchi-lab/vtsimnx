### solver/tests_cpp（C++ユニットテスト）

このフォルダは **C++ solver の最小ユニットテスト**です。

- 目的: 数式/パーサ/軽量ユーティリティの回帰を、solver本体の巨大ケース実行に頼らず素早く検出する
- 方針: 依存を増やさないため、現状は **自前の最小テストハーネス（`main()` + expect関数）**で実装しています

---

### 実行方法（ctest）

リポジトリルートで:

```bash
cmake -S solver -B build-solver -DCMAKE_BUILD_TYPE=Release -DVTSIMNX_BUILD_CPP_TESTS=ON
cmake --build build-solver -j
ctest --test-dir build-solver --output-on-failure
```

---

### 追加すると効果が大きいテストの例

- **パーサ**: `response_conduction` の必須項目・配列長チェック（JSON入力の破壊的変更を早期検出）
- **数式**: 換気流量の偶奇性/連続性/ヤコビアン整合（既存: `test_flow_math.cpp`）
- **安定化**: 係数の境界条件やフォールバック条件（例: sum(c)≈1）を「関数単位」で固定化



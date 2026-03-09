### VTSimNX API

このリポジトリは VTSimNX の **Python（FastAPI）ラッパ + builder + C++ solver** を含みます。

---

### 使い方

- FastAPI 起動: `RUN_FASTAPI.md`
- Python テスト: `tests_py/README.md`
- C++ solver テスト:
  - 通常: `solver/` 配下で `cmake --build build-solver && ctest --test-dir build-solver`
  - **低負荷（推奨: SSH切断/CPU飽和を避けたい場合）**:
    - ビルド並列を制限: `cmake --build build-solver -j1`（または `-j2`）
    - テスト並列を制限: `ctest --test-dir build-solver -j1 --output-on-failure`
- builder入力JSON: `docs/builder_json.md`
- シミュレーション全体の概略: `docs/simulation_overview.md`
- エアコンモデル（acmodel）: `docs/acmodel_overview.md`
- エアコン制御（ON/OFF・能力上限制御）: `docs/aircon_control_overview.md`
- 物理・数学メモ: `docs/physics_math_notes.md`

---

### 熱の壁モデル（RC / 応答係数法）

壁の層モデルは builder で生成します。

- **RC法（従来）**: 層ノードを作って RC ネットワークで解く
- **応答係数法（CTF）**: `response_conduction` ブランチで壁伝熱を表現（壁蓄熱により両面熱流が一致しないケースを扱える）

入力仕様と単位の約束は以下を参照してください:

- `docs/thermal_response_factor.md`
- `docs/thermal_rc.md`



---

### GitHub とローカルの対応

- このディレクトリ `/home/ubuntu/vtsimnx/api` は **単独の Git リポジトリ** で、GitHub 上の [`iguchi-lab/vtsimnx-api`](https://github.com/iguchi-lab/vtsimnx-api) に対応します。
- デフォルトブランチ: `main` （`origin/main` と追従）
- **規則**:
  - API 側のコード・builder・C++ solver（このリポジトリ配下の `solver/`）の変更は、必ずこのリポジトリで `git commit` `git push` します。
  - 親ディレクトリ `/home/ubuntu/vtsimnx` には Git 管理を置かず、`git` コマンドはこのディレクトリ（または `core/` 側）から実行します。
  - Python コアライブラリ（`vtsimnx` パッケージ）側の変更は `/home/ubuntu/vtsimnx/core` リポジトリで管理します（別リポジトリ）。

### VTSimNX API

このリポジトリは VTSimNX の **Python（FastAPI）ラッパ + builder + C++ solver** を含みます。

---

### 使い方

- FastAPI 起動: `RUN_FASTAPI.md`
- Python テスト: `tests_py/README.md`
- builder入力JSON: `docs/builder_json.md`
- 物理・数学メモ: `docs/physics_math_notes.md`

---

### 熱の壁モデル（RC / 応答係数法）

壁の層モデルは builder で生成します。

- **RC法（従来）**: 層ノードを作って RC ネットワークで解く
- **応答係数法（CTF）**: `response_conduction` ブランチで壁伝熱を表現（壁蓄熱により両面熱流が一致しないケースを扱える）

入力仕様と単位の約束は以下を参照してください:

- `docs/thermal_response_factor.md`
- `docs/thermal_rc.md`



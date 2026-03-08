### docs

- `docs/theory_basics.md`: 建築環境工学の基礎（換気回路網/熱回路網/日射・放射/湿気・濃度）と、本実装での扱い方
- `docs/builder_json.md`: builder入力JSON（raw_config）の作り方（トップレベル構造、key記法、時系列、surface/aircon展開、validation）
- `docs/simulation_overview.md`: シミュレーション全体の概略（builder→solver、1タイムステップの処理順、出力artifact）
- `docs/acmodel_overview.md`: エアコンモデル（acmodel）の概要（モデル種別、ac_spec、solver入力、出力）
- `docs/aircon_control_overview.md`: エアコン制御の概要（ON/OFF判定、fixed-row、処理熱量、能力上限超過時の設定温度補正）
- `docs/physics_math_notes.md`: 物理・数学メモ（符号/単位/離散化/安定性の注意点）
- `docs/thermal_rc.md`: 壁モデル（RC法）
- `docs/thermal_response_factor.md`: 壁モデル（応答係数法/CTF, `response_conduction`）

---

### 読み方ガイド（目的別）

- 初めて使う: `docs/theory_basics.md` → `docs/simulation_overview.md` → `docs/builder_json.md`
- 入力JSONを作る: `docs/builder_json.md` → `docs/thermal_rc.md` / `docs/thermal_response_factor.md`
- 実装寄りに理解する: `docs/simulation_overview.md` → `docs/aircon_control_overview.md` → `docs/acmodel_overview.md`

---

### 低負荷でビルド/テストしたいとき（メモ）

CPU/RAMを抑えたい場合は並列数を下げます:

- build: `cmake --build build-solver -j1`（または `-j2`）
- test: `ctest --test-dir build-solver -j1 --output-on-failure`



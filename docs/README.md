### docs

- `docs/builder_json.md`: builder入力JSON（raw_config）の作り方（トップレベル構造、key記法、時系列、surface/aircon展開、validation）
- `docs/simulation_overview.md`: シミュレーション全体の概略（builder→solver、1タイムステップの処理順、出力artifact）
- `docs/acmodel_overview.md`: エアコンモデル（acmodel）の概要（モデル種別、ac_spec、solver入力、出力）
- `docs/physics_math_notes.md`: 物理・数学メモ（符号/単位/離散化/安定性の注意点）
- `docs/thermal_rc.md`: 壁モデル（RC法）
- `docs/thermal_response_factor.md`: 壁モデル（応答係数法/CTF, `response_conduction`）

---

### 低負荷でビルド/テストしたいとき（メモ）

CPU/RAMを抑えたい場合は並列数を下げます:

- build: `cmake --build build-solver -j1`（または `-j2`）
- test: `ctest --test-dir build-solver -j1 --output-on-failure`



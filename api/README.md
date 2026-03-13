# VTSimNX API

VTSimNX のシミュレーションを HTTP で実行するための FastAPI ラッパです。  
`/run` に JSON を送ると、builder で設定を正規化した後に C++ solver を実行し、結果と artifact 情報を返します。

## この API でできること

- `GET /ping`: ヘルスチェック
- `POST /run`: シミュレーション実行
- `GET /artifacts/...`: 実行後の `schema.json` / `solver.log` / バイナリ結果の取得

詳細仕様は `docs/api_reference.md` を参照してください。

## クイックスタート

1. API を起動

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. 動作確認

```bash
curl -sS http://127.0.0.1:8000/ping
# {"status":"ok"}
```

3. シミュレーション実行

```bash
curl -sS -X POST http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "simulation": {"step": 1, "timestep": 3600},
      "nodes": [{"key": "outside"}, {"key": "room"}],
      "ventilation_branches": [],
      "thermal_branches": []
    }
  }'
```

レスポンス例（抜粋）:

```json
{
  "result": {
    "artifact_dir": "run_20260312_xxxxxxxx",
    "result_files": {
      "schema": "schema.json"
    }
  },
  "warnings": [],
  "warning_details": []
}
```

## ドキュメント

- API仕様: `docs/api_reference.md`
- 起動・運用メモ: `RUN_FASTAPI.md`
- builder 入力仕様: `docs/builder_json.md`
- シミュレーション全体像: `docs/simulation_overview.md`
- 空調モデル概要: `docs/acmodel_overview.md`
- 空調制御仕様: `docs/aircon_control_overview.md`
- 湿気回路網（Phase1）: `docs/moisture_network_phase1.md`
- 物理・数学メモ: `docs/physics_math_notes.md`

## 開発者向け情報

- 参加方法・テスト・コミット方針: `CONTRIBUTING.md`
- リポジトリ運用メモ: `docs/developer_notes.md`

## License

This project is licensed under the MIT License. See `LICENSE`.

## Disclaimer

- This software is provided for research and development purposes.  
  本ソフトウェアは研究・開発目的で提供されます。
- Results from simulation are not guaranteed to be accurate, complete, or fit for any specific purpose.  
  シミュレーション結果の正確性、完全性、特定目的への適合性は保証されません。
- Users are responsible for validating inputs, assumptions, and outputs before operational use.  
  運用利用前の入力条件・仮定・出力結果の妥当性確認は利用者の責任で実施してください。
- The authors and contributors are not liable for any direct or indirect damages arising from use of this software.  
  本ソフトウェアの利用に起因する直接的または間接的な損害について、著者および貢献者は責任を負いません。
- The legally binding warranty and liability terms are defined by `LICENSE` (MIT License).  
  法的拘束力を持つ保証・責任条件は `LICENSE`（MIT License）に従います。

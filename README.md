# vtsimnx
熱・換気回路網計算による温熱シミュレーションプログラム

## クイックスタート

1) 仮想環境の作成・有効化（Windows PowerShell）

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

2) 依存のインストール

```powershell
python -m pip install --upgrade pip
pip install -e .[dev]
# astropy を使う機能/テストも有効にする場合
# pip install -e .[dev,astro]
```

3) テストを実行

```powershell
python -m pytest
```

4) サンプル実行

```powershell
python run.py
```

`run.py` は `3639999.has` を読み込み、日射・表面設定・ノード/分岐などの入力データを作って、`run_calc` 経由でAPI（`/run`）に渡します（builderはAPI側で実行されます）。

補足:
- `vt.schedule` にスケジュール一式（aircon/vol/sensible_heat/latent_moisture）をまとめています
- `vt.materials` は材料物性テーブル（dict）です
- ドキュメント一覧は `docs/README.md` を参照してください
- 建築環境工学の基礎解説は `docs/building_environment_engineering_basics.md` を参照してください
- 日射計算 API (`solar_gain_by_angles` / `solar_gain_by_angles_with_shade`) の使い方は `docs/solar_usage.md` を参照してください
- 風圧/夜間放射/快適性 API (`make_wind` / `nocturnal_gain_by_angles` / `calc_PMV` など) の使い方は `docs/archenv_comfort_nocturnal_wind_usage.md` を参照してください

APIサーバーを使用する場合は、環境変数 `VTSIMNX_API_URL` を設定してください（例: `VTSIMNX_API_URL=http://localhost:8000`）。未設定の場合、`run.py` は入力生成のみ行い `run_calc` をスキップします。

※ `hasp/lat/lon` を変更したい場合は、現在は `run.py` 内の固定値（`hasp_path`, `lat`, `lon`）を編集してください。

---

### リポジトリ構成（モノレポ）

- 本リポジトリは `core`（Python ライブラリ）と `api/`（FastAPI + builder + C++ solver）を同居させたモノレポです。
- デフォルトブランチは `main` です（`origin/main` 追従）。
- **運用の境界**:
  - 外部利用者向けのライブラリ公開・利用対象は `core` 側（`vtsimnx` パッケージ / `vt.*` モジュール）。
  - 計算サーバーとして配備する対象は `api/` 側。
  - 仕様変更が両者にまたがる場合は、同一PR/同一コミット系列で整合をとります。

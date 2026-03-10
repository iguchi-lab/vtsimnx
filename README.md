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

### GitHub とローカルの対応

- このディレクトリ `/home/ubuntu/vtsimnx/core` は **単独の Git リポジトリ** で、GitHub 上の [`iguchi-lab/vtsimnx`](https://github.com/iguchi-lab/vtsimnx) に対応します。
- デフォルトブランチ: `main` （`origin/main` と追従）
- **規則**:
  - Python コアライブラリ（`vtsimnx` パッケージ / `vt.*` モジュール）の変更は、必ずこのリポジトリで `git commit` `git push` します。
  - API サーバー（FastAPI + builder + C++ solver）は別リポジトリ `/home/ubuntu/vtsimnx/api` （`iguchi-lab/vtsimnx-api`）で管理し、API 側の変更はそちらで commit/push します。
  - 親ディレクトリ `/home/ubuntu/vtsimnx` 自体は Git 管理せず、`core/` と `api/` を **2つの独立したリポジトリ** として扱います。

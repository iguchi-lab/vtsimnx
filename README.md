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

APIサーバーを使用する場合は、環境変数 `VTSIMNX_API_URL` を設定してください（例: `VTSIMNX_API_URL=http://localhost:8000`）。未設定の場合、`run.py` は入力生成のみ行い `run_calc` をスキップします。

※ `hasp/lat/lon` を変更したい場合は、現在は `run.py` 内の固定値（`hasp_path`, `lat`, `lon`）を編集してください。
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

`run.py` は `3639999.has` を読み込み、日射・表面設定・ノード/分岐を作って `build_config` を呼びます。引数で緯度経度やファイルパスを変更可能です（`python run.py --lat 35.68 --lon 139.77 --hasp 3639999.has`）。

補足: `build_config` はデフォルトではファイル出力しません。JSONを保存したい場合は `output_path` を明示してください（例: `vt.build_config(..., output_path="parsed_input_data.json.gz")`）。
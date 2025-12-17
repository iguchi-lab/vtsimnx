"""
C++ 製 VTSimNX ソルバの実行を担う薄いラッパーモジュール。

- 入力 JSON をファイルへ書き出す
- ソルバ実行ファイルを `subprocess.run` で起動する
- 生成された出力 JSON を読み戻して Python の辞書にして返す
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

# プロジェクトルート（このファイルの親の親）を基準にパスを解決する。
BASE_DIR = Path(__file__).resolve().parent.parent
# 想定するソルバ実行ファイルのパス。トップレベル build/ を参照する。
# 絶対パスで解決して、どのディレクトリから実行しても正しく動作するようにする。
SOLVER_EXE = (BASE_DIR / "build" / "vtsimnx_solver").resolve()


def _invoke_solver(input_path: Path, output_path: Path, cwd: Path) -> None:
    """
    共通のソルバ実行ロジック。
    subprocess.run の設定やエラーハンドリングを一箇所に集約する。
    """
    result = subprocess.run(
        [str(SOLVER_EXE), str(input_path), str(output_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"solver failed: {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(f"solver did not produce output file: {output_path}")

def run_solver(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    入力辞書を一時 JSON ファイルに書き出して C++ ソルバを実行し、
    生成された出力 JSON を辞書として返す。

    処理の流れ:
        1. `work/` ディレクトリ配下に `input.json` を作成
        2. 既存の `output.json` を削除（古い結果の混入防止）
        3. ソルバを `subprocess.run` で起動（カレントは `work/`）
        4. `output.json` を読み取って Python の辞書にして返却

    Args:
        input_data: ソルバに渡す設定内容（JSON 互換の辞書）

    Returns:
        Dict[str, Any]: ソルバが出力した JSON の内容

    Raises:
        RuntimeError: ソルバが異常終了した、または出力が生成されなかった場合
    """
    work_dir = BASE_DIR / "work"
    work_dir.mkdir(exist_ok=True)

    input_path = work_dir / "input.json"
    output_path = work_dir / "output.json"

    # 入力を書き出し
    with input_path.open("w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=2)

    # 既存の出力ファイルがあれば削除
    if output_path.exists():
        output_path.unlink()

    _invoke_solver(input_path, output_path, cwd=work_dir)

    with output_path.open("r", encoding="utf-8") as f:
        output_data = json.load(f)

    return output_data

def run_solver_from_files(input_path: "Path | str", output_path: "Path | str") -> Dict[str, Any]:
    """
    テスト/スクリプト用: 既存の入力ファイルと出力ファイルパスを受け取り、
    C++ ソルバを実行して出力 JSON を返す。

    Args:
        input_path: 入力 JSON ファイルのパス（相対パスまたは絶対パス）
        output_path: ソルバに書き出させる出力 JSON ファイルのパス（相対パスまたは絶対パス）

    Returns:
        Dict[str, Any]: 出力 JSON の内容

    Raises:
        RuntimeError: ソルバが異常終了した、または出力が生成されなかった場合
    """
    # パスを解決（相対パスは現在の作業ディレクトリを基準に解決）
    in_path = Path(input_path)
    out_path = Path(output_path)
    
    # 絶対パスに変換（相対パスの場合は現在の作業ディレクトリを基準に解決）
    if not in_path.is_absolute():
        in_abs = Path.cwd() / in_path
    else:
        in_abs = in_path
        
    if not out_path.is_absolute():
        out_abs = Path.cwd() / out_path
    else:
        out_abs = out_path
    
    # 絶対パスに正規化
    in_abs = in_abs.resolve()
    out_abs = out_abs.resolve()

    # 出力先ディレクトリを作成
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    if out_abs.exists():
        out_abs.unlink()
    _invoke_solver(in_abs, out_abs, cwd=out_abs.parent)

    with out_abs.open("r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VTSimNX solver")
    parser.add_argument("input_path", type=str, help="Input JSON file path")
    parser.add_argument("output_path", type=str, help="Output JSON file path")
    args = parser.parse_args()
    result = run_solver_from_files(args.input_path, args.output_path)
    # 出力は C++ 側が <output>.json と artifact_dir 配下にまとめて書き出す。
    # ここでは重複ファイル（.results.json 等）を作らず、output.json の生成のみで統一する。
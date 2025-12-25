"""
C++ 製 VTSimNX ソルバの実行を担う薄いラッパーモジュール。

- 入力 JSON をファイルへ書き出す
- ソルバ実行ファイルを `subprocess.run` で起動する
- 生成された出力 JSON を読み戻して Python の辞書にして返す
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import os

# プロジェクトルート（このファイルの親の親）を基準にパスを解決する。
BASE_DIR = Path(__file__).resolve().parent.parent
# 想定するソルバ実行ファイルのパス。トップレベル build/ を参照する。
# 絶対パスで解決して、どのディレクトリから実行しても正しく動作するようにする。
SOLVER_EXE = (BASE_DIR / "build" / "vtsimnx_solver").resolve()

def force_log_verbosity(config: Dict[str, Any], *, debug: bool, debug_verbosity: int, default_verbosity: int = 1) -> None:
    """
    API/CLI 共通: ログ冗長度を統制する。
    - debug=false: 常に verbosity=default_verbosity に落とす（指定があっても上書き）
    - debug=true : verbosity を debug_verbosity まで引き上げ（既に高い場合は維持）
    """
    sim = config.get("simulation")
    if not isinstance(sim, dict):
        sim = {}
        config["simulation"] = sim
    log = sim.get("log")
    if not isinstance(log, dict):
        log = {}
        sim["log"] = log

    if debug:
        try:
            current = int(log.get("verbosity", 0))
        except Exception:
            current = 0
        log["verbosity"] = max(current, int(debug_verbosity))
    else:
        log["verbosity"] = int(default_verbosity)

def set_log_verbosity(config: Dict[str, Any], verbosity: int) -> None:
    """API/CLI 共通: verbosity を明示的にセットする。"""
    sim = config.get("simulation")
    if not isinstance(sim, dict):
        sim = {}
        config["simulation"] = sim
    log = sim.get("log")
    if not isinstance(log, dict):
        log = {}
        sim["log"] = log
    log["verbosity"] = int(verbosity)

def _artifact_dir_from_output(work_dir: Path, output_data: Dict[str, Any]) -> Optional[Path]:
    """
    C++ ソルバが返す output.json の `artifact_dir` から、work_dir 配下の artifact パスを解決する。
    """
    artifact_dir = output_data.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    artifact_dir_path = (work_dir / artifact_dir).resolve()

    # セキュリティ: work_dir 外を指していないことを確認（パストラバーサル対策）
    work_root = work_dir.resolve()
    if work_root not in artifact_dir_path.parents and artifact_dir_path != work_root:
        return None
    return artifact_dir_path

def write_artifact_manifest(output_data: Dict[str, Any]) -> Optional[Path]:
    """
    artifact_dir 配下に manifest.json を保存する。
    - artifact取得APIのホワイトリスト/メタ情報として使う
    - work/output.json は上書きされ得るので、artifact側に固定で残す
    """
    work_dir = BASE_DIR / "work"
    artifact_dir_path = _artifact_dir_from_output(work_dir, output_data)
    if artifact_dir_path is None:
        return None
    artifact_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir_path / "manifest.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": output_data,
    }
    # UTF-8で確実に保存（ログ本文など巨大データは入れない想定）
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


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

    # 並列実行安全性:
    # 同一 work_dir で input.json / output.json を共有すると、同時リクエストで上書き競合が起きる。
    # そのため、リクエストごとにユニークなファイル名を使う。
    run_id = uuid.uuid4().hex
    input_path = work_dir / f"input.{run_id}.json"
    output_path = work_dir / f"output.{run_id}.json"

    # 入力を書き出し
    with input_path.open("w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=2)

    keep_run_files = os.getenv("VTSIMNX_KEEP_RUN_FILES") is not None
    try:
        _invoke_solver(input_path, output_path, cwd=work_dir)

        with output_path.open("r", encoding="utf-8") as f:
            output_data = json.load(f)
    finally:
        # デフォルトでは一時入出力を消して work/ 汚染を抑える（必要なら env で残せる）
        if not keep_run_files:
            try:
                if input_path.exists():
                    input_path.unlink()
            except Exception:
                pass
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

    # artifact_dir 配下に manifest を残す（後続のダウンロードAPIで参照）
    try:
        write_artifact_manifest(output_data)
    except Exception:
        # manifest書き込み失敗は致命ではないので握りつぶす（ログ/運用で気づけるようにするなら後で改善）
        pass

    return output_data
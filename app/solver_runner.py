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
import hashlib
import tempfile
import shutil

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

def attach_builder_log_to_artifacts(
    output_data: Dict[str, Any],
    *,
    builder_log_path: Path,
    artifact_filename: str = "builder.log",
    delete_source: bool = False,
) -> Optional[Path]:
    """
    builder のログファイルを artifacts 配下にコピーし、output_data に参照キーを追加する。
    - API の download は artifact_dir 直下しか許可していないため、サブディレクトリは使わない。

    追加するキー:
      output_data["builder_log_file"] = artifact_filename
    """
    if not builder_log_path or not isinstance(builder_log_path, Path):
        return None
    if not builder_log_path.exists() or not builder_log_path.is_file():
        return None

    work_dir = BASE_DIR / "work"
    artifact_dir_path = _artifact_dir_from_output(work_dir, output_data)
    if artifact_dir_path is None:
        return None
    artifact_dir_path.mkdir(parents=True, exist_ok=True)

    dest = artifact_dir_path / artifact_filename
    try:
        shutil.copy2(builder_log_path, dest)
    except Exception:
        return None
    finally:
        if delete_source:
            try:
                builder_log_path.unlink()
            except Exception:
                pass

    output_data["builder_log_file"] = artifact_filename
    return dest


def _invoke_solver(input_path: Path, output_path: Path, cwd: Path) -> None:
    """
    共通のソルバ実行ロジック。
    subprocess.run の設定やエラーハンドリングを一箇所に集約する。
    環境変数 VTSIMNX_SOLVER_TIMEOUT（秒・正の整数）を設定すると、その秒数で打ち切る。
    未設定または 0 の場合はタイムアウトなし。
    """
    timeout_s: Optional[int] = None
    try:
        raw = os.getenv("VTSIMNX_SOLVER_TIMEOUT", "").strip()
        if raw:
            t = int(raw)
            if t > 0:
                timeout_s = t
    except ValueError:
        pass

    try:
        result = subprocess.run(
            [str(SOLVER_EXE), str(input_path), str(output_path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"solver timed out after {e.timeout}s. "
            "Increase VTSIMNX_SOLVER_TIMEOUT or optimize the run."
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"solver failed: {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(f"solver did not produce output file: {output_path}")

class _HashingWriter:
    """
    json.dump の出力をファイルへ書きつつ、同じバイト列でハッシュ（sha256）も計算する。
    - 巨大JSONのためにメモリへ全体を保持しない
    """
    def __init__(self, f, h: "hashlib._Hash"):
        self._f = f
        self._h = h

    def write(self, s: str) -> int:
        b = s.encode("utf-8")
        self._h.update(b)
        return self._f.write(s)

    def flush(self) -> None:
        return self._f.flush()

def _write_input_json(
    input_data: Dict[str, Any],
    *,
    path: Path,
    pretty: bool,
    sort_keys: bool,
) -> None:
    """
    solver 入力JSONを書き出す。
    - デフォルトは compact（indentなし, separators指定）でサイズと parse 時間を削減
    - pretty はデバッグ用（KEEP_RUN_FILES と併用されがち）
    """
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(input_data, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        else:
            json.dump(input_data, f, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)

def _get_cached_input_path(
    work_dir: Path,
    input_data: Dict[str, Any],
    *,
    pretty: bool,
) -> Path:
    """
    入力JSONの内容に基づいてキャッシュファイルを返す（無ければ作成）。
    - キャッシュを使うことで「同一入力の連続実行」で I/O と JSON parse の両方が効く
    - ハッシュは json.dump の出力バイト列（UTF-8）に対して計算する
    """
    cache_dir = work_dir / "cache_inputs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 一旦テンポラリへ書き、同時に sha256 を計算 → hash が確定したら cache に move
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(work_dir), delete=False, prefix="input.cache.tmp.", suffix=".json") as tf:
        tmp_path = Path(tf.name)
        hw = _HashingWriter(tf, h)
        if pretty:
            json.dump(input_data, hw, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(input_data, hw, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        tf.write("\n")
        tf.flush()

    digest = h.hexdigest()
    cached = cache_dir / f"{digest}.json"

    try:
        if cached.exists():
            # 既にあるなら tmp を捨てる
            try:
                tmp_path.unlink()
            except Exception:
                pass
            return cached
        tmp_path.replace(cached)
        return cached
    finally:
        # replace が失敗した場合でも tmp が残る可能性があるのでベストエフォートで削除
        if tmp_path.exists() and cached.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

def run_solver(
    input_data: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    write_manifest: bool = True,
) -> Dict[str, Any]:
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
    run_id = (run_id or uuid.uuid4().hex)
    output_path = work_dir / f"output.{run_id}.json"

    keep_run_files = os.getenv("VTSIMNX_KEEP_RUN_FILES") is not None
    pretty_input = os.getenv("VTSIMNX_PRETTY_INPUT") is not None
    # KEEP_RUN_FILES のときは読みやすさ優先で pretty に寄せる（明示指定があればそちら優先）
    if keep_run_files and os.getenv("VTSIMNX_PRETTY_INPUT") is None:
        pretty_input = True

    use_input_cache = os.getenv("VTSIMNX_INPUT_CACHE") is not None

    # 入力を書き出し（またはキャッシュから取得）
    cached_input = False
    if use_input_cache:
        input_path = _get_cached_input_path(work_dir, input_data, pretty=pretty_input)
        cached_input = True
    else:
        input_path = work_dir / f"input.{run_id}.json"
        _write_input_json(input_data, path=input_path, pretty=pretty_input, sort_keys=False)

    try:
        _invoke_solver(input_path, output_path, cwd=work_dir)

        with output_path.open("r", encoding="utf-8") as f:
            output_data = json.load(f)
    finally:
        # デフォルトでは一時入出力を消して work/ 汚染を抑える（必要なら env で残せる）
        if not keep_run_files:
            try:
                # キャッシュ入力は消さない
                if (not cached_input) and input_path.exists():
                    input_path.unlink()
            except Exception:
                pass
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

    # artifact_dir 配下に manifest を残す（後続のダウンロードAPIで参照）
    if write_manifest:
        try:
            write_artifact_manifest(output_data)
        except Exception:
            # manifest書き込み失敗は致命ではないので握りつぶす（ログ/運用で気づけるようにするなら後で改善）
            pass

    return output_data
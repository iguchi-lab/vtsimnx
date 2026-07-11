"""
C++ 製 VTSimNX ソルバの実行を担う薄いラッパーモジュール。

- 入力 JSON をファイルへ書き出す
- ソルバ実行ファイルを `subprocess.run` で起動する
- 生成された出力 JSON を読み戻して Python の辞書にして返す
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid
import os
import hashlib
import tempfile
import shutil
import threading
import time as time_mod

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


def run_workdir(run_id: str) -> Path:
    """1 run 専用の作業ディレクトリ: work/runs/{run_id}/"""
    rid = (run_id or "").strip()
    if not rid or "/" in rid or "\\" in rid or ".." in rid:
        raise ValueError(f"invalid run_id: {run_id!r}")
    path = BASE_DIR / "work" / "runs" / rid
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_dir_completeness_score(path: Path) -> int:
    """solver 成果物が揃っているディレクトリを優先するための簡易スコア。"""
    score = 0
    try:
        if (path / "schema.json").is_file():
            score += 100
        if (path / "solver.log").is_file():
            score += 50
        score += sum(1 for _ in path.glob("*.f32.bin"))
    except OSError:
        return 0
    return score


def resolve_artifact_path(artifact_dir: str) -> Optional[Path]:
    """
    artifact_dir（basename）を work/ 直下または work/runs/*/ 配下から解決する。

    同名ディレクトリが複数ある場合は、schema.json / solver.log / f32.bin が
    揃っている方を優先する（builder.log だけ置いた空シェルを誤選択しない）。
    """
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    if "/" in artifact_dir or "\\" in artifact_dir or ".." in artifact_dir:
        return None

    work_root = (BASE_DIR / "work").resolve()
    candidates: list[Path] = []
    direct = (work_root / artifact_dir).resolve()
    if work_root in direct.parents and direct.is_dir():
        candidates.append(direct)

    runs_root = work_root / "runs"
    if runs_root.is_dir():
        for candidate in runs_root.glob(f"*/{artifact_dir}"):
            resolved = candidate.resolve()
            if work_root in resolved.parents and resolved.is_dir():
                candidates.append(resolved)
    if not candidates:
        return None
    return max(candidates, key=_artifact_dir_completeness_score)


def cleanup_run_workdir(run_id: str, *, keep_artifacts: bool = True) -> None:
    """
    当該 run の一時ファイルだけを削除する（他 run には触れない）。
    keep_artifacts=True のとき artifacts.* ディレクトリは残す。
    """
    try:
        rid = (run_id or "").strip()
        if not rid or "/" in rid or "\\" in rid or ".." in rid:
            return
        run_dir = (BASE_DIR / "work" / "runs" / rid).resolve()
        work_runs = (BASE_DIR / "work" / "runs").resolve()
        if work_runs not in run_dir.parents and run_dir != work_runs:
            return
        if not run_dir.exists():
            return

        if not keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
            return

        for pattern in ("input.*", "output.*", "builder.log.tmp", "input.cache.tmp.*"):
            for p in run_dir.glob(pattern):
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def _artifact_dir_from_output(work_dir: Path, output_data: Dict[str, Any]) -> Optional[Path]:
    """
    C++ ソルバが返す output.json の `artifact_dir` から、work_dir 配下の artifact パスを解決する。
    work_dir 直下に無ければ work/runs/*/ も検索する。

    注意: 未作成パスを返さない（mkdir で空シェルを作ってしまうのを防ぐ）。
    """
    artifact_dir = output_data.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None

    artifact_dir_path = (work_dir / artifact_dir).resolve()
    work_root = work_dir.resolve()
    if artifact_dir_path.is_dir() and (
        work_root in artifact_dir_path.parents or artifact_dir_path == work_root
    ):
        return artifact_dir_path

    # run 隔離後: attach/manifest が共有 work/ を渡しても見つかるようにする
    return resolve_artifact_path(artifact_dir)


def attach_log_tail_to_output(
    output_data: Dict[str, Any],
    *,
    max_chars: int = 4000,
) -> Optional[str]:
    """
    失敗時に solver.log 末尾を result.log.text へ埋め込み、リモートでも即読めるようにする。
    """
    if not isinstance(output_data, dict):
        return None

    artifact_dir_name = output_data.get("artifact_dir")
    if isinstance(artifact_dir_name, str) and artifact_dir_name:
        artifact_dir_path = resolve_artifact_path(artifact_dir_name)
    else:
        artifact_dir_path = None
    if artifact_dir_path is None:
        work_dir = BASE_DIR / "work"
        artifact_dir_path = _artifact_dir_from_output(work_dir, output_data)
    if artifact_dir_path is None:
        return None

    log_file = output_data.get("log_file")
    if not isinstance(log_file, str) or not log_file:
        log_file = "solver.log"
    log_path = artifact_dir_path / log_file
    if not log_path.is_file():
        return None

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]

    log_obj = output_data.get("log")
    if not isinstance(log_obj, dict):
        log_obj = {}
        output_data["log"] = log_obj
    log_obj["text"] = text
    return text


def write_artifact_manifest(output_data: Dict[str, Any]) -> Optional[Path]:
    """
    artifact_dir 配下に manifest.json を保存する。
    - artifact取得APIのホワイトリスト/メタ情報として使う
    - work/output.json は上書きされ得るので、artifact側に固定で残す
    """
    artifact_dir_name = output_data.get("artifact_dir")
    if isinstance(artifact_dir_name, str) and artifact_dir_name:
        artifact_dir_path = resolve_artifact_path(artifact_dir_name)
    else:
        artifact_dir_path = None
    if artifact_dir_path is None:
        work_dir = BASE_DIR / "work"
        artifact_dir_path = _artifact_dir_from_output(work_dir, output_data)
    if artifact_dir_path is None:
        return None
    artifact_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_dir_path / "manifest.json"
    result_files = output_data.get("result_files")
    if not isinstance(result_files, dict):
        result_files = {}

    # 互換:
    # 一部クライアントは manifest.json のトップレベル files/result_files を参照する。
    # solver がエラーで result_files が空でも、最低限 log / builder_log / manifest へ辿れるよう
    # files は常に非空になり得るマップとして構成する。
    compat_files = dict(result_files)
    if isinstance(output_data.get("log_file"), str) and output_data.get("log_file"):
        compat_files["log"] = output_data["log_file"]
    if isinstance(output_data.get("builder_log_file"), str) and output_data.get("builder_log_file"):
        compat_files["builder_log"] = output_data["builder_log_file"]
    compat_files["manifest"] = "manifest.json"

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": output_data,
        # 互換:
        # - result_files: 従来どおり solver の結果ファイル群
        # - files: ログ/manifest を含む広いファイルマップ
        "result_files": result_files,
        "files": compat_files,
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
    build_stats: Optional[Tuple[int, int, int]] = None,
    build_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    builder のログファイルを artifacts 配下にコピーし、output_data に参照キーを追加する。
    - コピー前に「ビルド結果: ノード=〇, 熱ブランチ=〇, 換気ブランチ=〇」を1行追記する。
      build_stats が渡されていればそれを使い、なければ build_config から件数を算出する。
    - API の download は artifact_dir 直下しか許可していないため、サブディレクトリは使わない。

    追加するキー:
      output_data["builder_log_file"] = artifact_filename
    """
    if not builder_log_path or not isinstance(builder_log_path, Path):
        return None
    if not builder_log_path.exists() or not builder_log_path.is_file():
        return None

    if build_stats is None and build_config is not None:
        try:
            build_stats = (
                len(build_config.get("nodes") or []),
                len(build_config.get("thermal_branches") or []),
                len(build_config.get("ventilation_branches") or []),
            )
        except Exception:
            pass

    if build_stats is not None:
        try:
            n_nodes, n_thermal, n_vent = build_stats
            msg = "ビルド結果: ノード=%d, 熱ブランチ=%d, 換気ブランチ=%d" % (n_nodes, n_thermal, n_vent)
            with open(builder_log_path, "a", encoding="utf-8") as f:
                f.write("%s [INFO] %s\n" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], msg))
        except (ValueError, OSError):
            pass

    # ソルバが作った実ディレクトリを使う（work/ 直下に空の artifacts.* を新規作成しない）
    artifact_dir_name = output_data.get("artifact_dir")
    artifact_dir_path = None
    if isinstance(artifact_dir_name, str) and artifact_dir_name:
        artifact_dir_path = resolve_artifact_path(artifact_dir_name)
    if artifact_dir_path is None:
        artifact_dir_path = _artifact_dir_from_output(BASE_DIR / "work", output_data)
    if artifact_dir_path is None or not artifact_dir_path.is_dir():
        return None

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


_ACTIVE_SOLVERS: Dict[str, subprocess.Popen] = {}
_ACTIVE_SOLVERS_LOCK = threading.Lock()


def terminate_solver(run_id: str) -> bool:
    """実行中の solver 子プロセスを terminate する。"""
    with _ACTIVE_SOLVERS_LOCK:
        proc = _ACTIVE_SOLVERS.get(run_id)
    if proc is None:
        return False
    try:
        proc.terminate()
        return True
    except Exception:
        return False


def _invoke_solver(
    input_path: Path,
    output_path: Path,
    cwd: Path,
    *,
    run_id: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    共通のソルバ実行ロジック。
    subprocess の設定やエラーハンドリングを一箇所に集約する。
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

    proc = subprocess.Popen(
        [str(SOLVER_EXE), str(input_path), str(output_path)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if run_id:
        with _ACTIVE_SOLVERS_LOCK:
            _ACTIVE_SOLVERS[run_id] = proc

    stdout = ""
    stderr = ""
    t0 = time_mod.perf_counter()
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    raise RuntimeError("solver cancelled")
                if timeout_s is not None and (time_mod.perf_counter() - t0) >= timeout_s:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"solver timed out after {timeout_s}s. "
                        "Increase VTSIMNX_SOLVER_TIMEOUT or optimize the run."
                    )
    finally:
        if run_id:
            with _ACTIVE_SOLVERS_LOCK:
                _ACTIVE_SOLVERS.pop(run_id, None)

    if proc.returncode != 0:
        # C++ 側が writeErrorOutput 済みなら、構造化エラーを呼び出し側で読めるようにする。
        if output_path.exists():
            return
        raise RuntimeError(
            f"solver failed: {proc.returncode}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
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
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    入力辞書を一時 JSON ファイルに書き出して C++ ソルバを実行し、
    生成された出力 JSON を辞書として返す。

    処理の流れ:
        1. `work/runs/{run_id}/` に input/output を作成
        2. ソルバを subprocess で起動（カレントは run 専用ディレクトリ）
        3. output JSON を読み取って Python の辞書にして返却
    """
    shared_work = BASE_DIR / "work"
    shared_work.mkdir(exist_ok=True)

    run_id = (run_id or uuid.uuid4().hex)
    run_dir = run_workdir(run_id)
    output_path = run_dir / f"output.{run_id}.json"

    keep_run_files = os.getenv("VTSIMNX_KEEP_RUN_FILES") is not None
    pretty_input = os.getenv("VTSIMNX_PRETTY_INPUT") is not None
    if keep_run_files and os.getenv("VTSIMNX_PRETTY_INPUT") is None:
        pretty_input = True

    use_input_cache = os.getenv("VTSIMNX_INPUT_CACHE") is not None

    cached_input = False
    if use_input_cache:
        input_path = _get_cached_input_path(shared_work, input_data, pretty=pretty_input)
        cached_input = True
    else:
        input_path = run_dir / f"input.{run_id}.json"
        _write_input_json(input_data, path=input_path, pretty=pretty_input, sort_keys=False)

    try:
        _invoke_solver(
            input_path,
            output_path,
            cwd=run_dir,
            run_id=run_id,
            cancel_event=cancel_event,
        )

        with output_path.open("r", encoding="utf-8") as f:
            output_data = json.load(f)
    finally:
        if not keep_run_files:
            try:
                if (not cached_input) and input_path.exists():
                    input_path.unlink()
            except Exception:
                pass
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

    if write_manifest:
        try:
            write_artifact_manifest(output_data)
        except Exception:
            pass

    return output_data

"""
FastAPI ベースの VTSimNX API エンドポイント定義。

- /ping: ライブネス/ヘルスチェック
- /run: C++ ソルバを呼び出して結果 JSON を返す
"""
#
# 注意: `python3 app/main.py ...` のように「パッケージ配下のファイルをスクリプト実行」すると、
# sys.path の先頭が `.../app/` になり `import app` が失敗する（app/app を探してしまう）。
# デバッグ目的の単発実行では `python3 -m app.main ...` を推奨するが、
# 互換のためスクリプト実行時はリポジトリルートを sys.path に追加して動作させる。
import sys
from pathlib import Path as _Path

if __name__ == "__main__" and (globals().get("__package__") in (None, "")):
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path
import json
import gzip
import os
import logging
import uuid
import tempfile
import time
from app.solver_runner import run_solver, force_log_verbosity
from app.solver_runner import attach_builder_log_to_artifacts, write_artifact_manifest
from app.builder import build_config_with_warning_details
from app.builder.validate import ValidationError, ConfigFileError
from app.builder.logger import use_builder_log_file, cleanup_default_work_logs

# Uvicorn のロガー設定に追従して出す（traceback を残すため）
logger = logging.getLogger(__name__)

# API ルータ本体。OpenAPI のタイトルやバージョンをここで設定する。
app = FastAPI(title="VTSimNX API", version="1.0.5")

BASE_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = BASE_DIR / "work"

class GZipRequestMiddleware:
    """
    Content-Encoding: gzip のとき、リクエストボディを展開して下流へ渡す。
    - body: gzip decompress → UTF-8 JSON（FastAPI/Pydanticは通常どおり処理できる）
    - Response: 現状どおりJSON（レスポンス圧縮は任意）
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        enc_raw = headers.get(b"content-encoding", b"").decode("latin1").lower()
        # "gzip" または "gzip, something" のような複合指定も許可
        enc_tokens = [t.strip() for t in enc_raw.split(",") if t.strip()]
        if "gzip" not in enc_tokens and "x-gzip" not in enc_tokens:
            return await self.app(scope, receive, send)

        # リクエストボディを全部読む（gzipはストリーム展開より簡単・確実を優先）
        max_compressed = int(os.getenv("VTSIMNX_MAX_GZIP_BODY_BYTES", str(64 * 1024 * 1024)))  # 64MiB
        max_decompressed = int(os.getenv("VTSIMNX_MAX_JSON_BODY_BYTES", str(256 * 1024 * 1024)))  # 256MiB
        body = b""
        more_body = True
        try:
            while more_body:
                msg = await receive()
                if msg["type"] != "http.request":
                    continue
                body += msg.get("body", b"")
                if max_compressed > 0 and len(body) > max_compressed:
                    resp = JSONResponse({"detail": "gzip body too large"}, status_code=413)
                    return await resp(scope, receive, send)
                more_body = msg.get("more_body", False)
            decompressed = gzip.decompress(body)
            if max_decompressed > 0 and len(decompressed) > max_decompressed:
                resp = JSONResponse({"detail": "decompressed body too large"}, status_code=413)
                return await resp(scope, receive, send)
        except Exception:
            resp = JSONResponse({"detail": "invalid gzip body"}, status_code=400)
            return await resp(scope, receive, send)

        # 下流へは展開済みボディを1回だけ返す
        async def receive2():
            return {"type": "http.request", "body": decompressed, "more_body": False}

        # Content-Encoding/Lengthを整合させる
        new_headers = []
        for k, v in (scope.get("headers") or []):
            if k in (b"content-encoding", b"content-length"):
                continue
            new_headers.append((k, v))
        new_headers.append((b"content-length", str(len(decompressed)).encode("ascii")))
        scope["headers"] = new_headers

        return await self.app(scope, receive2, send)

app.add_middleware(GZipRequestMiddleware)

def _resolve_artifact_dir(artifact_dir: str) -> Path:
    # パストラバーサル防止: work配下のディレクトリに限定
    if "/" in artifact_dir or "\\" in artifact_dir or ".." in artifact_dir:
        raise HTTPException(status_code=400, detail="invalid artifact_dir")
    p = (WORK_DIR / artifact_dir).resolve()
    work_root = WORK_DIR.resolve()
    if work_root not in p.parents:
        raise HTTPException(status_code=400, detail="invalid artifact_dir")
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail="artifact_dir not found")
    return p

def _load_manifest(artifact_path: Path) -> Dict[str, Any]:
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest.json not found (run /run once to generate it)")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read manifest.json: {e}")

def _artifact_file_from_key(manifest: Dict[str, Any], key: str) -> Tuple[str, str]:
    """
    manifest(output.json相当) からキーに対応するファイル名を返す。
    戻り値: (filename, media_type)
    """
    out = manifest.get("output", {})
    if not isinstance(out, dict):
        raise HTTPException(status_code=500, detail="invalid manifest format")

    if key == "log":
        name = out.get("log_file")
        if not isinstance(name, str) or not name:
            raise HTTPException(status_code=404, detail="log_file not available")
        return name, "text/plain"

    if key == "builder_log":
        name = out.get("builder_log_file")
        if not isinstance(name, str) or not name:
            raise HTTPException(status_code=404, detail="builder_log_file not available")
        return name, "text/plain"

    if key == "manifest":
        return "manifest.json", "application/json"

    result_files = out.get("result_files", {})
    if not isinstance(result_files, dict):
        raise HTTPException(status_code=404, detail="result_files not available")
    name = result_files.get(key)
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=404, detail=f"unknown file key: {key}")

    if name.endswith(".json"):
        return name, "application/json"
    if name.endswith(".log"):
        return name, "text/plain"
    return name, "application/octet-stream"

def _build_bad_request_detail(e: Exception) -> Dict[str, Any]:
    """
    入力不正(400)のレスポンス本文を構造化して返す。
    クライアントが機械的に扱えるよう code/message を固定化する。
    """
    message = str(e)
    code = "invalid_config"
    detail: Dict[str, Any] = {}

    # KeyError は "'outside'" のような表示になりがちなので補足して返す
    if isinstance(e, KeyError):
        missing = str(e.args[0]) if getattr(e, "args", None) else message
        code = "invalid_config_missing_field"
        message = f"必須フィールド '{missing}' が不足しています。"
        detail["hint"] = f"入力JSONに '{missing}' を追加してください。"

    detail.update({
        "code": code,
        "message": message,
    })

    # 典型的なノード参照ミスには修正ヒントを添える
    if "ノード" in message and "存在しません" in message:
        detail["hint"] = "nodes に参照先ノードを追加するか、参照先の key を既存ノード名に合わせてください。"
    if "ventilated layer" in message and "requires positive 't'" in message:
        detail["hint"] = "ventilated_air_layer=true の層には正の厚さ t（例: 0.04）を指定してください。"
    return detail


def _build_internal_error_detail(e: Exception, *, run_id: str | None = None) -> Dict[str, Any]:
    """
    内部エラー(500)のレスポンス本文を構造化して返す。
    """
    detail: Dict[str, Any] = {
        "code": "internal_error",
        "message": str(e),
    }
    if run_id:
        detail["run_id"] = run_id

    if isinstance(e, FileNotFoundError) and "vtsimnx_solver" in str(e):
        detail["code"] = "solver_binary_not_found"
        detail["hint"] = "サーバ上で C++ solver 実行ファイル build/vtsimnx_solver をビルドしてください。"
    elif isinstance(e, RuntimeError) and str(e).startswith("solver failed:"):
        detail["code"] = "solver_execution_failed"
        detail["hint"] = "solver.log と入力JSONを確認し、設定値や境界条件の不整合を見直してください。"

    return detail


class SimulationRequest(BaseModel):
    """
    ソルバに渡す入力設定を表すデータモデル。

    - config: ユーザー入力JSON（raw）。API側で `app.builder.build_config()` により
      正規化/展開してから C++ ソルバに渡す。
    """
    config: Dict[str, Any]
    debug: bool = False
    debug_verbosity: int = 2
    # builder オプション（APIから制御）
    # None の場合は raw_config["builder"]（または builder 側既定値）に従う
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None

class SimulationResponse(BaseModel):
    """
    ソルバの計算結果を表すデータモデル。

    - result: ソルバから返却される任意の JSON 互換オブジェクト
    """
    result: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    warning_details: List[Dict[str, Any]] = Field(default_factory=list)


def _attach_api_timings(output: Dict[str, Any], api_timings: Dict[str, float]) -> None:
    """
    /run の API レイヤ時間内訳を result に埋め込む。
    既存キーを壊さないよう、`api_timings` 配下へ追加する。
    """
    output["api_timings"] = api_timings

@app.get("/ping")
def ping():
    """軽量なヘルスチェック用エンドポイント。"""
    return {"status": "ok"}

@app.post("/run", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest):
    """
    入力 JSON（config）を C++ ソルバに渡して、結果 JSON を返す。

    Args:
        req: クライアントから受け取った `SimulationRequest`。`config` を含む。

    Returns:
        `SimulationResponse`: ソルバの出力を `result` に格納して返す。

    Raises:
        HTTPException(500): ソルバ実行や結果読み取りで例外が発生した場合に返す。
    """
    run_id: str | None = None
    api_t0 = time.perf_counter()
    try:
        # 1リクエスト=1 run_id を先に決める（builderログとsolver成果物を紐付けるため）
        run_id = uuid.uuid4().hex

        # builder ログは /tmp に一時出力（work/logs には出さない）。
        # その後、solver の artifacts が確定したら artifacts 直下へコピーして manifest に載せ、元の一時ファイルは削除する。
        tmp_dir = os.getenv("VTSIMNX_BUILDER_TMP_DIR") or tempfile.gettempdir()
        builder_log_tmp = Path(tmp_dir) / f"vtsimnx.builder.{run_id}.log"

        # ユーザー入力（raw）を solver 用 config に変換（正規化/展開/検証）
        build_stats_out: list = []
        builder_t0 = time.perf_counter()
        with use_builder_log_file(builder_log_tmp):
            try:
                built_config, warnings, warning_details = build_config_with_warning_details(
                    req.config,
                    output_path=None,
                    add_surface=req.add_surface,
                    add_aircon=req.add_aircon,
                    add_capacity=req.add_capacity,
                    add_moisture_capacity=req.add_moisture_capacity,
                    add_surface_solar=req.add_surface_solar,
                    add_surface_nocturnal=req.add_surface_nocturnal,
                    add_surface_radiation=req.add_surface_radiation,
                    add_surface_radiation_exclude_glass=req.add_surface_radiation_exclude_glass,
                    build_stats_out=build_stats_out,
                )
            except (ValidationError, ConfigFileError, ValueError, KeyError, TypeError) as e:
                logger.info("validation/config error in /run: %s", e)
                raise HTTPException(status_code=400, detail=_build_bad_request_detail(e))
        builder_t1 = time.perf_counter()

        # API側でログ冗長度を統制（debug=falseなら常に1に落とす）
        # ※ build_config の後に適用することで、builder の未知キー削除で落ちないようにする
        force_log_verbosity(built_config, debug=req.debug, debug_verbosity=req.debug_verbosity, default_verbosity=1)

        # manifest は builder_log を追記してから書きたいので、一旦書かずに返してもらう。
        # テストでは run_solver を「引数1個のlambda」でモックしているので、kwargs が使えない場合はフォールバックする。
        solver_t0 = time.perf_counter()
        try:
            output = run_solver(built_config, run_id=run_id, write_manifest=False)
        except TypeError:
            # モック想定（本番のsolverでここに落ちるのは基本的に想定しない）
            output = run_solver(built_config)
        solver_t1 = time.perf_counter()

        # builderログを artifacts へ取り込み、manifest(output) に参照を追加（ビルド結果行は attach 内で追記）
        artifact_t0 = time.perf_counter()
        attach_builder_log_to_artifacts(
            output,
            builder_log_path=builder_log_tmp,
            artifact_filename="builder.log",
            delete_source=True,
            build_config=built_config,
        )
        write_artifact_manifest(output)
        artifact_t1 = time.perf_counter()

        api_t1 = time.perf_counter()
        api_timings = {
            "builder_ms": (builder_t1 - builder_t0) * 1000.0,
            "solver_ms": (solver_t1 - solver_t0) * 1000.0,
            "artifact_postprocess_ms": (artifact_t1 - artifact_t0) * 1000.0,
            "api_total_ms": (api_t1 - api_t0) * 1000.0,
        }
        _attach_api_timings(output, api_timings)
    except (ValidationError, ConfigFileError) as e:
        # 入力不正は 400
        logger.info("validation/config error in /run: %s", e)
        raise HTTPException(status_code=400, detail=_build_bad_request_detail(e))
    except HTTPException:
        raise
    except Exception as e:
        # エラー時は 500 を返す
        logger.exception("internal error in /run")
        raise HTTPException(status_code=500, detail=_build_internal_error_detail(e, run_id=run_id))
    finally:
        # 既定の work/logs は一次置き場として扱い、都度クリーンアップする。
        cleanup_default_work_logs()

    return SimulationResponse(result=output, warnings=warnings, warning_details=warning_details)


def _run_simulation_core(
    *,
    raw_config: Dict[str, Any],
    debug: bool,
    debug_verbosity: int,
    add_surface: Optional[bool] = None,
    add_aircon: Optional[bool] = None,
    add_capacity: Optional[bool] = None,
    add_moisture_capacity: Optional[bool] = None,
    add_surface_solar: Optional[bool] = None,
    add_surface_nocturnal: Optional[bool] = None,
    add_surface_radiation: Optional[bool] = None,
    add_surface_radiation_exclude_glass: Optional[bool] = None,
) -> SimulationResponse:
    """
    /run と同じ経路で単発実行したいときの共通ロジック（CLI/テスト用）。
    FastAPI の依存注入や HTTP レイヤに依存しない。
    """
    try:
        api_t0 = time.perf_counter()
        run_id = uuid.uuid4().hex
        tmp_dir = os.getenv("VTSIMNX_BUILDER_TMP_DIR") or tempfile.gettempdir()
        builder_log_tmp = Path(tmp_dir) / f"vtsimnx.builder.{run_id}.log"
        build_stats_out: list = []
        builder_t0 = time.perf_counter()
        with use_builder_log_file(builder_log_tmp):
            built_config, warnings, warning_details = build_config_with_warning_details(
                raw_config,
                output_path=None,
                add_surface=add_surface,
                add_aircon=add_aircon,
                add_capacity=add_capacity,
                add_moisture_capacity=add_moisture_capacity,
                add_surface_solar=add_surface_solar,
                add_surface_nocturnal=add_surface_nocturnal,
                add_surface_radiation=add_surface_radiation,
                add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
                build_stats_out=build_stats_out,
            )
        builder_t1 = time.perf_counter()
        force_log_verbosity(built_config, debug=debug, debug_verbosity=debug_verbosity, default_verbosity=1)
        solver_t0 = time.perf_counter()
        output = run_solver(built_config, run_id=run_id, write_manifest=False)
        solver_t1 = time.perf_counter()
        artifact_t0 = time.perf_counter()
        attach_builder_log_to_artifacts(
            output,
            builder_log_path=builder_log_tmp,
            artifact_filename="builder.log",
            delete_source=True,
            build_config=built_config,
        )
        write_artifact_manifest(output)
        artifact_t1 = time.perf_counter()
        api_t1 = time.perf_counter()
        api_timings = {
            "builder_ms": (builder_t1 - builder_t0) * 1000.0,
            "solver_ms": (solver_t1 - solver_t0) * 1000.0,
            "artifact_postprocess_ms": (artifact_t1 - artifact_t0) * 1000.0,
            "api_total_ms": (api_t1 - api_t0) * 1000.0,
        }
        _attach_api_timings(output, api_timings)
        return SimulationResponse(result=output, warnings=warnings, warning_details=warning_details)
    finally:
        # CLI/テスト経路でも work/logs の一時ログを残さない。
        cleanup_default_work_logs()

@app.get("/artifacts/{artifact_dir}/manifest")
def get_artifact_manifest(artifact_dir: str):
    """
    artifact_dir 配下の manifest.json を返す。
    """
    artifact_path = _resolve_artifact_dir(artifact_dir)
    return _load_manifest(artifact_path)

@app.get("/artifacts/{artifact_dir}/files")
def list_artifact_files(artifact_dir: str):
    """
    ダウンロード可能なファイルキー一覧を返す（ホワイトリスト）。
    """
    artifact_path = _resolve_artifact_dir(artifact_dir)
    manifest = _load_manifest(artifact_path)
    out = manifest.get("output", {})
    result_files = out.get("result_files", {}) if isinstance(out, dict) else {}
    keys = []
    if isinstance(result_files, dict):
        keys.extend(sorted([k for k, v in result_files.items() if isinstance(v, str) and v]))
    if isinstance(out, dict) and isinstance(out.get("log_file"), str) and out.get("log_file"):
        keys.append("log")
    if isinstance(out, dict) and isinstance(out.get("builder_log_file"), str) and out.get("builder_log_file"):
        keys.append("builder_log")
    keys.append("manifest")
    return {"artifact_dir": artifact_dir, "keys": keys}

@app.get("/artifacts/{artifact_dir}/download/{key}")
def download_artifact_file(artifact_dir: str, key: str):
    """
    ファイル本体を返す（巨大データは FileResponse によりストリーミング送信される）。
    key は /artifacts/{artifact_dir}/files で得たもののみ許可する。
    """
    artifact_path = _resolve_artifact_dir(artifact_dir)
    manifest = _load_manifest(artifact_path)
    filename, media_type = _artifact_file_from_key(manifest, key)

    # パストラバーサル防止: artifact_dir直下のみ許可
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename in manifest")

    file_path = (artifact_path / filename).resolve()
    if artifact_path not in file_path.parents:
        raise HTTPException(status_code=400, detail="invalid file path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)

# 静的ファイル配信: プロジェクト直下の work/ を /work に配信
app.mount("/work", StaticFiles(directory=str(WORK_DIR)), name="work")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run VTSimNX once (same path as API /run)")
    parser.add_argument("input_path", type=str, help="Input JSON file path (raw config)")
    parser.add_argument("--output", type=str, default=None, help="Write SimulationResponse JSON to this path")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--debug", action="store_true", help="デバッグ: verbosity を引き上げる（最低 debug_verbosity）")
    g.add_argument("--quiet", action="store_true", help="静かに: verbosity=0（silent）にする")
    parser.add_argument("--debug-verbosity", type=int, default=2, help="--debug時のverbosity下限（既定: 2）")
    parser.add_argument("--verbosity", type=int, default=None, help="verbosityを明示指定（指定時は--debug/--quietより優先）")
    args = parser.parse_args()

    try:
        raw = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("input.json root must be object")
        if args.verbosity is not None:
            # 明示指定が最優先
            sim = raw.get("simulation")
            if not isinstance(sim, dict):
                sim = {}
                raw["simulation"] = sim
            log = sim.get("log")
            if not isinstance(log, dict):
                log = {}
                sim["log"] = log
            log["verbosity"] = int(args.verbosity)
            debug = True  # builder後に force_log_verbosity が上書きしないよう debug扱いにする
            debug_verbosity = int(args.verbosity)
        elif args.quiet:
            sim = raw.get("simulation")
            if not isinstance(sim, dict):
                sim = {}
                raw["simulation"] = sim
            log = sim.get("log")
            if not isinstance(log, dict):
                log = {}
                sim["log"] = log
            log["verbosity"] = 0
            debug = True
            debug_verbosity = 0
        else:
            debug = bool(args.debug)
            debug_verbosity = int(args.debug_verbosity)

        resp = _run_simulation_core(raw_config=raw, debug=debug, debug_verbosity=debug_verbosity)
        payload = resp.model_dump()
        text = json.dumps(payload, ensure_ascii=False, indent=2)

        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
    except (ValidationError, ConfigFileError) as e:
        sys.stderr.write(str(e) + "\n")
        raise SystemExit(2)
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        raise SystemExit(1)

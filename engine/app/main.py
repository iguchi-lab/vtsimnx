"""
FastAPI ベースの VTSimNX API エンドポイント定義。

- /ping: ライブネス/ヘルスチェック
- /run: 同期シミュレーション（互換）
- /runs: 非同期ジョブ API
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

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
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
from contextlib import asynccontextmanager

from app.solver_runner import resolve_artifact_path, run_solver
from app.simulation import execute_simulation
from app.builder import build_config_with_warning_details
from app.builder.validate import ValidationError, ConfigFileError
from app.api_auth import ApiKeyMiddleware
from app.jobs import get_run_manager, set_run_manager, RunManager
from app import simulation as _simulation_mod


def _sync_test_hooks() -> None:
    """テストが main 上のシンボルを差し替えた場合に simulation へ反映する。"""
    _simulation_mod._run_solver_hook = run_solver
    _simulation_mod.build_config_with_warning_details = build_config_with_warning_details
# Uvicorn のロガー設定に追従して出す（traceback を残すため）
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    manager = RunManager()
    set_run_manager(manager)
    try:
        yield
    finally:
        manager.shutdown(wait=False)
        set_run_manager(None)


# API ルータ本体。OpenAPI のタイトルやバージョンをここで設定する。
app = FastAPI(title="VTSimNX API", version="1.0.8", lifespan=_lifespan)

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
app.add_middleware(ApiKeyMiddleware)

def _resolve_artifact_dir(artifact_dir: str) -> Path:
    # パストラバーサル防止: basename のみ。work/ 直下または work/runs/*/ を検索
    if "/" in artifact_dir or "\\" in artifact_dir or ".." in artifact_dir:
        raise HTTPException(status_code=400, detail="invalid artifact_dir")
    p = resolve_artifact_path(artifact_dir)
    if p is None:
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
    elif isinstance(e, RuntimeError) and str(e).startswith("solver timed out"):
        detail["code"] = "solver_timeout"
        detail["hint"] = "VTSIMNX_SOLVER_TIMEOUT を延ばすか、計算条件（期間・分割・連成）を見直してください。"
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
    入力 JSON（config）を C++ ソルバに渡して、結果 JSON を返す（同期・互換API）。
    長時間計算は POST /runs を推奨。
    """
    run_id: str | None = None
    try:
        _sync_test_hooks()

        result = execute_simulation(
            req.config,
            debug=req.debug,
            debug_verbosity=req.debug_verbosity,
            add_surface=req.add_surface,
            add_aircon=req.add_aircon,
            add_capacity=req.add_capacity,
            add_moisture_capacity=req.add_moisture_capacity,
            add_surface_solar=req.add_surface_solar,
            add_surface_nocturnal=req.add_surface_nocturnal,
            add_surface_radiation=req.add_surface_radiation,
            add_surface_radiation_exclude_glass=req.add_surface_radiation_exclude_glass,
        )
        run_id = result.run_id
        return SimulationResponse(
            result=result.output,
            warnings=result.warnings,
            warning_details=result.warning_details,
        )
    except (ValidationError, ConfigFileError, ValueError, KeyError, TypeError) as e:
        logger.info("validation/config error in /run: %s", e)
        raise HTTPException(status_code=400, detail=_build_bad_request_detail(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("internal error in /run")
        raise HTTPException(status_code=500, detail=_build_internal_error_detail(e, run_id=run_id))


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
    """
    _sync_test_hooks()
    result = execute_simulation(
        raw_config,
        debug=debug,
        debug_verbosity=debug_verbosity,
        add_surface=add_surface,
        add_aircon=add_aircon,
        add_capacity=add_capacity,
        add_moisture_capacity=add_moisture_capacity,
        add_surface_solar=add_surface_solar,
        add_surface_nocturnal=add_surface_nocturnal,
        add_surface_radiation=add_surface_radiation,
        add_surface_radiation_exclude_glass=add_surface_radiation_exclude_glass,
    )
    return SimulationResponse(
        result=result.output,
        warnings=result.warnings,
        warning_details=result.warning_details,
    )


@app.post("/runs")
def create_run(req: SimulationRequest, response: Response):
    """非同期ジョブを投入し、run_id を即時返す。重複入力は既存 run_id を 200 で返す。"""
    manager = get_run_manager()
    _sync_test_hooks()
    request = req.model_dump()
    job, is_duplicate = manager.submit(request)
    response.status_code = 200 if is_duplicate else 202
    return {
        "run_id": job.run_id,
        "status": job.status,
        "input_hash": job.input_hash,
    }


@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    job = get_run_manager().get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")
    return job.to_status_dict()


@app.get("/runs/{run_id}/result", response_model=SimulationResponse)
def get_run_result(run_id: str):
    job = get_run_manager().get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")
    if job.status == "succeeded" and isinstance(job.result, dict):
        return SimulationResponse(**job.result)
    if job.status in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail={"code": "not_ready", "message": f"run status is {job.status}", "run_id": run_id},
        )
    if job.status == "cancelled":
        raise HTTPException(
            status_code=409,
            detail={"code": "cancelled", "message": "run was cancelled", "run_id": run_id, **(job.error or {})},
        )
    raise HTTPException(
        status_code=500,
        detail=job.error or {"code": "internal_error", "message": "run failed", "run_id": run_id},
    )


@app.delete("/runs/{run_id}")
def cancel_run(run_id: str):
    job = get_run_manager().cancel(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")
    return job.to_status_dict()


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

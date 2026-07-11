import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import vtsimnx as vt


ARTIFACT_DIR = "output.artifacts.123"


class _State:
    post_run = 0
    get_artifacts = 0
    get_schema = 0
    get_manifest = 0
    get_bin = 0
    get_log = 0


def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        _State.post_run += 1

        # Windows環境だと、POSTボディを読まずに応答すると接続中断扱いになることがある。
        # ここで Content-Length 分を読み捨てておく。
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n > 0:
            _ = self.rfile.read(n)

        # 結果（/run レスポンス）: log.text があるので log 取得はHTTP不要にできる
        body = {
            "result": {
                "artifact_dir": ARTIFACT_DIR,
                "log_file": "solver.log",
                "log": {"text": "preloaded log"},
                "timings": [
                    {"name": "load_input", "duration_ms": 3.2},
                    {"name": "simulation_total", "duration_ms": 120.5},
                ],
                "result_files": {
                    "schema": "schema.json",
                    "vent_flow_rate": "vent.flow_rate.f32.bin",
                    "vent_pressure": "vent.pressure.f32.bin",
                },
            }
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        base = f"/artifacts/{ARTIFACT_DIR}"
        if self.path.startswith(base):
            _State.get_artifacts += 1

        if self.path == f"{base}/manifest":
            _State.get_manifest += 1
            manifest = {
                "output": {
                    "log_file": "solver.log",
                    "result_files": {
                        "schema": "schema.json",
                        "vent_flow_rate": "vent.flow_rate.f32.bin",
                        "vent_pressure": "vent.pressure.f32.bin",
                    },
                }
            }
            _send(self, 200, json.dumps(manifest).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"{base}/download/schema":
            _State.get_schema += 1
            schema = {
                "dtype": "f32le",
                "layout": "timestep-major",
                "length": 2,
                "series": {
                    "vent_flow_rate": {"keys": ["c1", "c2"]},
                    "vent_pressure": {"keys": ["p1"]},
                },
            }
            _send(self, 200, json.dumps(schema).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"{base}/download/vent_flow_rate":
            _State.get_bin += 1
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return

        if self.path == f"{base}/download/vent_pressure":
            _State.get_bin += 1
            arr = np.array([10.0, 20.0], dtype=np.dtype("<f4"))  # (T=2, N=1)
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return

        if self.path == f"{base}/download/log":
            _State.get_log += 1
            _send(self, 200, b"hello\n", "text/plain; charset=utf-8")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class _ErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n > 0:
            _ = self.rfile.read(n)

        body = {
            "result": {
                "artifact_dir": ARTIFACT_DIR,
                "status": "error",
                "error": "nodes[928].pre_temp must be array<number>",
                "log_file": "solver.log",
                "builder_log_file": "builder.log",
                "result_files": {},
            }
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format, *args):
        return


def test_run_calc_with_dataframes_is_lazy():
    _State.post_run = 0
    _State.get_artifacts = 0
    _State.get_schema = 0
    _State.get_manifest = 0
    _State.get_bin = 0
    _State.get_log = 0

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        res = vt.run_calc(
            base_url,
            {"simulation": {"index": {"length": 2, "timestep": 1}}},
            output_path=None,
            as_result=True,
            compress_request=False,  # このスタブはgzipを解凍しない,
                use_legacy_run=True)

        # /run 以外の GET は、まだ走っていない（遅延ロード）
        assert _State.post_run == 1
        assert _State.get_artifacts == 0
        assert hasattr(res, "get_series_df")
        assert isinstance(res.client_profile, dict)
        assert "run_calc_total_ms" in res.client_profile
        assert "run_post_ms" in res.client_profile

        # log はレスポンス内に埋まっているので GET なし
        assert res.log == "preloaded log"
        assert _State.get_log == 0

        server_timings = res.get_server_timings()
        assert len(server_timings) == 2
        report = res.get_timing_report()
        assert report["server"]["load_input_ms"] == pytest.approx(3.2)
        assert report["server"]["simulation_total_ms"] == pytest.approx(120.5)

        # DataFrame を要求したときだけ GET が走る
        df = res.get_series_df("vent_flow_rate")
        assert df is not None
        assert list(df.columns) == ["c1", "c2"]
        assert df.shape == (2, 2)
        assert "vent_flow_rate" in res.series_profiles
        assert _State.get_schema >= 1
        assert _State.get_bin >= 1

        # 別系列を要求しても schema は再取得しない（キャッシュ）
        schema_count = _State.get_schema
        df2 = res.get_series_df("vent_pressure")
        assert df2 is not None
        assert list(df2.columns) == ["p1"]
        assert df2.shape == (2, 1)
        assert _State.get_schema == schema_count
        assert _State.get_bin >= 2
    finally:
        server.shutdown()
        server.server_close()


def test_run_calc_with_dataframes_raises_original_error_message():
    server = HTTPServer(("127.0.0.1", 0), _ErrorHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        with pytest.raises(ValueError, match="nodes\\[928\\]\\.pre_temp must be array<number>"):
            vt.run_calc(
                base_url,
                {"simulation": {"index": {"length": 2, "timestep": 1}}},
                output_path=None,
                as_result=True,
                compress_request=False,
                use_legacy_run=True)
    finally:
        server.shutdown()
        server.server_close()

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import vtsimnx as vt
from vtsimnx.artifacts._schema import extract_manifest_error
from vtsimnx.run_calc import RunCalcAPIError


def test_extract_manifest_error_includes_artifact_and_log_tail():
    msg = extract_manifest_error(
        {
            "status": "error",
            "error": "coupling did not converge",
            "artifact_dir": "artifacts.demo",
            "log_file": "solver.log",
            "builder_log_file": "builder.log",
            "log": {"text": "....\nERROR: pressure residual\n"},
        }
    )
    assert msg is not None
    assert "coupling did not converge" in msg
    assert "artifact_dir=artifacts.demo" in msg
    assert "solver.log (tail)" in msg
    assert "pressure residual" in msg


def test_run_calc_raises_readable_api_error_on_400():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n > 0:
                self.rfile.read(n)
            body = {
                "detail": {
                    "code": "invalid_config",
                    "message": "ノード 'X' が存在しません",
                    "hint": "nodes に参照先ノードを追加してください。",
                }
            }
            out = json.dumps(body).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        with pytest.raises(RunCalcAPIError) as e:
            vt.run_calc(
                f"http://127.0.0.1:{port}",
                {"simulation": {"index": {"length": 1, "timestep": 1}}},
                with_dataframes=False,
                compress_request=False,
                use_legacy_run=True)
        assert e.value.status_code == 400
        assert "[invalid_config]" in str(e.value)
        assert "ノード 'X' が存在しません" in str(e.value)
        assert "hint:" in str(e.value)
    finally:
        server.shutdown()


def test_run_calc_raises_value_error_with_log_tail_on_sim_error():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n > 0:
                self.rfile.read(n)
            body = {
                "result": {
                    "status": "error",
                    "error": "did not converge",
                    "artifact_dir": "artifacts.1",
                    "log_file": "solver.log",
                    "log": {"text": "tail line\n"},
                    "result_files": {},
                }
            }
            out = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        with pytest.raises(ValueError) as e:
            vt.run_calc(
                f"http://127.0.0.1:{port}",
                {"simulation": {"index": {"length": 1, "timestep": 1}}},
                compress_request=False,
                use_legacy_run=True)
        assert "did not converge" in str(e.value)
        assert "artifact_dir=artifacts.1" in str(e.value)
        assert "tail line" in str(e.value)
    finally:
        server.shutdown()

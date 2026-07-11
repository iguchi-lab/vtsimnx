"""run_calc の /runs ポーリングクライアントのテスト。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import vtsimnx as vt
from vtsimnx.run_calc import RunCalcAPIError


class _PollState:
    posts = 0
    status_gets = 0
    result_gets = 0
    run_id = "abc123"
    fail_mode = None  # None | "failed" | "timeout"


def _make_handler():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            if n > 0:
                self.rfile.read(n)
            _PollState.posts += 1
            if self.path != "/runs":
                self.send_response(404)
                self.end_headers()
                return
            body = {
                "run_id": _PollState.run_id,
                "status": "queued",
                "input_hash": "0" * 64,
            }
            out = json.dumps(body).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            if self.path == f"/runs/{_PollState.run_id}":
                _PollState.status_gets += 1
                if _PollState.fail_mode == "failed":
                    status = "failed"
                    error = {"code": "internal_error", "message": "boom"}
                elif _PollState.fail_mode == "timeout":
                    status = "running"
                    error = None
                elif _PollState.status_gets >= 2:
                    status = "succeeded"
                    error = None
                else:
                    status = "running"
                    error = None
                body = {
                    "run_id": _PollState.run_id,
                    "status": status,
                    "input_hash": "0" * 64,
                    "progress": {"stage": status, "message": ""},
                    "error": error,
                }
                out = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            if self.path == f"/runs/{_PollState.run_id}/result":
                _PollState.result_gets += 1
                body = {
                    "result": {
                        "status": "ok",
                        "artifact_dir": "artifacts.poll",
                        "result_files": {},
                    },
                    "warnings": [],
                    "warning_details": [],
                }
                out = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    return _Handler


def _start_server():
    server = HTTPServer(("127.0.0.1", 0), _make_handler())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{port}"


def test_run_calc_poll_success():
    _PollState.posts = 0
    _PollState.status_gets = 0
    _PollState.result_gets = 0
    _PollState.fail_mode = None
    server, base = _start_server()
    try:
        out = vt.run_calc(
            base,
            {"simulation": {"index": {"length": 1, "timestep": 1}}},
            with_dataframes=False,
            compress_request=False,
            poll_interval=0.05,
            timeout=5.0,
        )
        assert out["result"]["status"] == "ok"
        assert _PollState.posts == 1
        assert _PollState.status_gets >= 2
        assert _PollState.result_gets == 1
    finally:
        server.shutdown()


def test_run_calc_poll_failed():
    _PollState.posts = 0
    _PollState.status_gets = 0
    _PollState.result_gets = 0
    _PollState.fail_mode = "failed"
    server, base = _start_server()
    try:
        with pytest.raises(RunCalcAPIError) as e:
            vt.run_calc(
                base,
                {"simulation": {"index": {"length": 1, "timestep": 1}}},
                with_dataframes=False,
                compress_request=False,
                poll_interval=0.05,
                timeout=5.0,
            )
        assert "failed" in str(e.value)
        assert _PollState.result_gets == 0
    finally:
        server.shutdown()


def test_run_calc_poll_timeout():
    _PollState.posts = 0
    _PollState.status_gets = 0
    _PollState.result_gets = 0
    _PollState.fail_mode = "timeout"
    server, base = _start_server()
    try:
        with pytest.raises(RunCalcAPIError) as e:
            vt.run_calc(
                base,
                {"simulation": {"index": {"length": 1, "timestep": 1}}},
                with_dataframes=False,
                compress_request=False,
                poll_interval=0.05,
                timeout=0.2,
            )
        assert "timed out" in str(e.value)
    finally:
        server.shutdown()

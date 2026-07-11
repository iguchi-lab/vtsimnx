"""run_calc の as_result / raise_on_error API 回帰。"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import vtsimnx as vt
from vtsimnx.artifacts.errors import ArtifactHTTPError, ArtifactNotFound
from vtsimnx.run_calc.run_calc import CalcRunResult, _resolve_as_result


ARTIFACT_DIR = "output.artifacts.api_clarity"


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
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n > 0:
            self.rfile.read(n)
        body = {
            "result": {
                "artifact_dir": ARTIFACT_DIR,
                "log_file": "solver.log",
                "result_files": {
                    "schema": "schema.json",
                    "vent_flow_rate": "vent.flow_rate.f32.bin",
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
        if self.path == f"{base}/manifest":
            manifest = {
                "output": {
                    "log_file": "solver.log",
                    "result_files": {
                        "schema": "schema.json",
                        "vent_flow_rate": "vent.flow_rate.f32.bin",
                    },
                }
            }
            _send(self, 200, json.dumps(manifest).encode("utf-8"), "application/json; charset=utf-8")
            return
        if self.path == f"{base}/download/schema":
            schema = {
                "dtype": "f32le",
                "layout": "timestep-major",
                "length": 2,
                "series": {"vent_flow_rate": {"keys": ["c1"]}},
            }
            _send(self, 200, json.dumps(schema).encode("utf-8"), "application/json; charset=utf-8")
            return
        if self.path == f"{base}/download/vent_flow_rate":
            arr = np.array([1.0, 2.0], dtype=np.dtype("<f4"))
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return
        if self.path == f"{base}/download/log":
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


@pytest.fixture
def server_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_resolve_as_result_defaults_and_conflicts():
    assert _resolve_as_result(as_result=None, with_dataframes=None) is True
    assert _resolve_as_result(as_result=False, with_dataframes=None) is False
    with pytest.warns(DeprecationWarning, match="with_dataframes"):
        assert _resolve_as_result(as_result=None, with_dataframes=False) is False
    with pytest.raises(ValueError, match="一致しません"):
        _resolve_as_result(as_result=True, with_dataframes=False)


def test_as_result_true_returns_calc_run_result(server_url):
    res = vt.run_calc(
        server_url,
        {"simulation": {"index": {"length": 2, "timestep": 1}}},
        as_result=True,
        compress_request=False,
        use_legacy_run=True,
    )
    assert isinstance(res, CalcRunResult)
    assert res.raise_on_error is False
    df = res.get_series_df("vent_flow_rate")
    assert df is not None
    assert df.shape == (2, 1)


def test_as_result_false_returns_dict(server_url):
    res = vt.run_calc(
        server_url,
        {"simulation": {"index": {"length": 2, "timestep": 1}}},
        as_result=False,
        compress_request=False,
        use_legacy_run=True,
    )
    assert isinstance(res, dict)
    assert "result" in res or "output" in res or "artifact_dir" in res


def test_with_dataframes_emits_deprecation_warning(server_url):
    with pytest.warns(DeprecationWarning, match="with_dataframes"):
        res = vt.run_calc(
            server_url,
            {"simulation": {"index": {"length": 2, "timestep": 1}}},
            with_dataframes=False,
            compress_request=False,
            use_legacy_run=True,
        )
    assert isinstance(res, dict)


def test_raise_on_error_missing_series(server_url):
    res = vt.run_calc(
        server_url,
        {"simulation": {"index": {"length": 2, "timestep": 1}}},
        as_result=True,
        raise_on_error=True,
        compress_request=False,
        use_legacy_run=True,
    )
    assert res.get_series_df("no_such_series", raise_on_error=False) is None
    with pytest.raises(ArtifactNotFound):
        res.get_series_df("no_such_series")


def test_raise_on_error_fetch_failure(server_url):
    res = vt.run_calc(
        server_url,
        {"simulation": {"index": {"length": 2, "timestep": 1}}},
        as_result=True,
        raise_on_error=False,
        compress_request=False,
        use_legacy_run=True,
    )
    assert res.get_log_text() is None
    assert "__log__" in res.errors

    res.raise_on_error = True
    with pytest.raises((ArtifactHTTPError, RuntimeError)):
        # _log_text は未設定のまま失敗パスへ（前回 None でキャッシュしていない）
        res._log_text = None
        res.errors.pop("__log__", None)
        res.get_log_text()

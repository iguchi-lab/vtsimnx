"""Resume a saved run through HTTP, including authenticated lazy artifacts."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

import vtsimnx as vt
from vtsimnx.run_calc import _http
from vtsimnx.tests.test_run_retry import Session, response


@pytest.mark.parametrize("from_environment", [False, True])
def test_resumed_result_matches_run_calc_without_resubmission(monkeypatch, tmp_path, from_environment):
    calls = []
    index = {"start": "2026-07-29T00:00:00", "length": 2, "timestep": 3600}
    output = {"result": {"status": "ok", "artifact_dir": "saved-artifacts", "index": index,
                         "log": {"text": "saved log"},
                         "result_files": {"schema": "schema.json", "thermal_temperature": "t.f32.bin"}}}
    schema = {"dtype": "f32le", "layout": "timestep-major", "length": 2,
              "series": {"thermal_temperature": {"keys": ["room"]}}}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append(("POST", self.path, self.headers.get("X-API-Key")))
            self.reply(200, json.dumps(output).encode())

        def do_GET(self):
            calls.append(("GET", self.path, self.headers.get("X-API-Key")))
            if self.headers.get("X-API-Key") != "test-resume-key":
                self.reply(401, b"{}")
            elif self.path == "/runs/saved-run":
                self.reply(200, b'{"status":"succeeded"}')
            elif self.path == "/runs/saved-run/result":
                self.reply(200, json.dumps(output).encode())
            elif self.path.endswith("/download/schema"):
                self.reply(200, json.dumps(schema).encode())
            elif self.path.endswith("/download/thermal_temperature"):
                self.reply(200, np.asarray([26., 26.5], dtype="<f4").tobytes())
            else:
                self.reply(404, b"{}")

        def log_message(self, *args):
            pass

    monkeypatch.setenv("VTSIMNX_API_KEY", "test-resume-key")
    key = None if from_environment else "test-resume-key"
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        original = vt.run_calc(base_url, {"simulation": {"index": index}},
                               use_legacy_run=True, api_key=key)
        saved = tmp_path / "result.json"
        resumed = vt.wait_for_run(base_url, "saved-run", as_result=True, api_key=key,
                                  raise_on_error=True, output_path=saved)
        assert isinstance(resumed, vt.CalcRunResult)
        assert resumed.config is None and resumed.raise_on_error
        assert resumed.output == original.output == json.loads(saved.read_text())
        assert resumed.client_profile["run_id"] == "saved-run"
        assert "wait_for_run_total_ms" in resumed.client_profile
        assert resumed.log == original.log == "saved log"
        assert len(calls) == 3  # No artifact requests before series access.
        got = resumed.get_series_df("thermal_temperature")
        pd.testing.assert_frame_equal(got, original.get_series_df("thermal_temperature"))
        assert list(got.iloc[:, 0]) == [26., 26.5]
        assert got.index.equals(pd.date_range(index["start"], periods=2, freq="h"))
        assert all(call[2] == "test-resume-key" for call in calls)
        assert sum(call[0] == "POST" for call in calls) == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("as_result", [None, False])
def test_resume_preserves_raw_response_default(monkeypatch, as_result):
    body = {"result": {"status": "ok"}, "warnings": ["saved warning"]}
    session = Session([response(200, {"status": "succeeded"}), response(200, body)])
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    kwargs = {} if as_result is None else {"as_result": as_result}
    assert vt.wait_for_run("http://test", "saved-run", **kwargs) == body
    assert session.closed and len(session.calls) == 2


@pytest.mark.parametrize("result", [{"status": "error", "error": "saved solver error"}, {"status": "ok"}])
def test_resume_result_rejects_error_or_missing_artifacts(monkeypatch, result):
    session = Session([response(200, {"status": "succeeded"}), response(200, {"result": result})])
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    with pytest.raises(ValueError):
        vt.wait_for_run("http://test", "saved-run", as_result=True)

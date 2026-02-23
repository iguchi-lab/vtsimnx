import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

import vtsimnx as vt


class _State:
    received = None


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        _State.received = json.loads(raw.decode("utf-8"))

        resp = {"result": {"artifact_dir": "output.artifacts.1", "result_files": {}}}
        out = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, format, *args):
        return


def test_run_calc_serializes_pandas_series():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"

        s = pd.Series([1.0, 2.0, None])
        idx = pd.date_range("2025-01-01 01:00:00", periods=3, freq="1h")
        cfg = {
            "simulation": {"index": idx},
            "nodes": [{"key": "外部", "t": s}],
        }

        _ = vt.run_calc(base_url, cfg, output_path=None, with_dataframes=False, compress_request=False)

        assert isinstance(_State.received, dict)
        sent_cfg = _State.received["config"]
        assert sent_cfg["nodes"][0]["t"] == [1.0, 2.0, None]
        assert sent_cfg["simulation"]["index"]["timestep"] == 3600
        assert sent_cfg["simulation"]["index"]["length"] == 3
    finally:
        server.shutdown()
        server.server_close()


def test_run_calc_does_not_mutate_input_config():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        idx = pd.date_range("2025-01-01 01:00:00", periods=3, freq="1h")
        cfg = {
            "simulation": {"index": idx},
            "nodes": [{"key": "外部", "t": pd.Series([1.0, 2.0, None])}],
        }

        _ = vt.run_calc(base_url, cfg, output_path=None, with_dataframes=False, compress_request=False)

        # run_calc 実行後も、呼び出し側の simulation.index は DatetimeIndex のまま
        assert isinstance(cfg["simulation"]["index"], pd.DatetimeIndex)
    finally:
        server.shutdown()
        server.server_close()


def test_run_calc_default_does_not_write_output_file(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        monkeypatch.chdir(tmp_path)
        base_url = f"http://127.0.0.1:{port}"
        cfg = {
            "simulation": {"index": pd.date_range("2025-01-01 01:00:00", periods=3, freq="1h")},
            "nodes": [{"key": "外部", "t": pd.Series([1.0, 2.0, None])}],
        }

        _ = vt.run_calc(base_url, cfg, with_dataframes=False, compress_request=False)
        assert not (tmp_path / "calc_result.json").exists()
    finally:
        server.shutdown()
        server.server_close()



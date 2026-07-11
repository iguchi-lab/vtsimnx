import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import vtsimnx as vt


ARTIFACT_DIR = "output.artifacts.123"


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

        # POSTボディを読み捨て（Windowsでの接続中断対策）
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n > 0:
            _ = self.rfile.read(n)

        body = {
            "output": {
                "artifact_dir": ARTIFACT_DIR,
                "index": {
                    "start": "2025-01-01 00:00:00",
                    "end": "2025-01-01 01:00:00",
                    "timestep": 3600,
                    "length": 2,
                },
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
                    "index": {
                        "start": "2025-01-01 00:00:00",
                        "end": "2025-01-01 01:00:00",
                        "timestep": 3600,
                        "length": 2,
                    },
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
                "series": {
                    "vent_flow_rate": {"keys": ["c1", "c2"]},
                },
            }
            _send(self, 200, json.dumps(schema).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"{base}/download/vent_flow_rate":
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def test_run_calc_get_series_df_prefers_output_index():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        # config側にも index を入れるが、output.index が優先されることを確認したい
        res = vt.run_calc(
            base_url,
            {
                "simulation": {
                    "index": {
                        "start": "1999-01-01 00:00:00",
                        "end": "1999-01-01 01:00:00",
                        "timestep": 3600,
                        "length": 2,
                    }
                }
            },
            output_path=None,
            as_result=True,
            compress_request=False,  # このスタブはgzipを解凍しない,
                use_legacy_run=True)

        df = res.get_series_df("vent_flow_rate")
        assert df is not None
        assert df.index.name == "time"
        assert str(df.index[0]) == "2025-01-01 00:00:00"
        assert str(df.index[1]) == "2025-01-01 01:00:00"
    finally:
        server.shutdown()
        server.server_close()

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import vtsimnx as vt


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
                "artifact_dir": "output.artifacts.123",
                "index": {
                    "start": "2025-01-01 00:00:00",
                    "end": "2025-01-01 01:00:00",
                    "timestep": 3600,
                    "length": 2,
                },
                "result_files": {
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
        if self.path == "/work/output.artifacts.123/schema.json":
            schema = {
                "dtype": "f32le",
                "layout": "timestep-major",
                "length": 2,
                "series": {
                    "vent_flow_rate": {"keys": ["c1", "c2"]},
                },
            }
            raw = json.dumps(schema).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path == "/work/output.artifacts.123/vent.flow_rate.f32.bin":
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            raw = arr.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
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
            with_dataframes=True,
            compress_request=False,  # このスタブはgzipを解凍しない
        )

        df = res.get_series_df("vent_flow_rate")
        assert df is not None
        assert df.index.name == "time"
        assert str(df.index[0]) == "2025-01-01 00:00:00"
        assert str(df.index[1]) == "2025-01-01 01:00:00"
    finally:
        server.shutdown()
        server.server_close()



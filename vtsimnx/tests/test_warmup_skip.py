import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import vtsimnx as vt
from vtsimnx.run_calc._index import _warmup_skip_rows


def _cfg(*, warmup, length=365, timestep=86400):
    sim = {
        "index": {
            "start": "2026-01-01 00:00:00",
            "end": "2026-12-31 00:00:00",
            "timestep": timestep,
            "length": length,
        }
    }
    if warmup is not None:
        sim["warmup"] = warmup
    return {"simulation": sim}


def test_skip_from_length_diff():
    cfg = _cfg(warmup={"start": "12-15"})
    assert _warmup_skip_rows(cfg, 365 + 17) == 17
    assert _warmup_skip_rows(cfg, 365) == 0


def test_skip_from_explicit_steps():
    cfg = _cfg(warmup={"start": "12-15", "steps": 17, "eval_length": 365})
    assert _warmup_skip_rows(cfg, 382) == 17


def test_skip_string_warmup():
    cfg = _cfg(warmup="12-15")
    assert _warmup_skip_rows(cfg, 382) == 17


def test_skip_without_warmup():
    assert _warmup_skip_rows(_cfg(warmup=None), 365) == 0
    assert _warmup_skip_rows({}, 365) == 0


def test_get_series_df_drops_warmup_rows():
    artifact_dir = "output.artifacts.warmup"

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n:
                self.rfile.read(n)
            body = {
                "output": {
                    "artifact_dir": artifact_dir,
                    "index": {
                        "start": "2025-12-30 00:00:00",
                        "end": "2026-01-03 00:00:00",
                        "timestep": 86400,
                        "length": 5,
                    },
                    "result_files": {
                        "schema": "schema.json",
                        "thermal_temperature": "thermal.temperature.f32.bin",
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
            base = f"/artifacts/{artifact_dir}"
            if self.path == f"{base}/manifest":
                manifest = {
                    "output": {
                        "index": {
                            "start": "2025-12-30 00:00:00",
                            "end": "2026-01-03 00:00:00",
                            "timestep": 86400,
                            "length": 5,
                        },
                        "result_files": {
                            "schema": "schema.json",
                            "thermal_temperature": "thermal.temperature.f32.bin",
                        },
                    }
                }
                raw = json.dumps(manifest).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == f"{base}/download/schema":
                schema = {
                    "dtype": "f32le",
                    "layout": "timestep-major",
                    "length": 5,
                    "series": {"thermal_temperature": {"keys": ["室"]}},
                }
                raw = json.dumps(schema).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == f"{base}/download/thermal_temperature":
                arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(arr.tobytes())))
                self.end_headers()
                self.wfile.write(arr.tobytes())
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        res = vt.run_calc(
            f"http://127.0.0.1:{port}",
            {
                "simulation": {
                    "index": {
                        "start": "2026-01-01 00:00:00",
                        "end": "2026-01-03 00:00:00",
                        "timestep": 86400,
                        "length": 3,
                    },
                    "warmup": {"start": "12-15"},
                }
            },
            output_path=None,
            as_result=True,
            compress_request=False,
            use_legacy_run=True,
        )
        df = res.get_series_df("thermal_temperature")
        assert df is not None
        assert df["室"].tolist() == [2.0, 3.0, 4.0]
        assert str(df.index[0]) == "2026-01-01 00:00:00"
        assert len(df) == 3
    finally:
        server.shutdown()
        server.server_close()

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
    def do_GET(self):
        if self.path == f"/artifacts/{ARTIFACT_DIR}/manifest":
            manifest = {
                "output": {
                    "log_file": "solver.log",
                    "result_files": {"schema": "schema.json"},
                },
                "files": {
                    "schema": "schema.json",
                    "log": "solver.log",
                    "manifest": "manifest.json",
                },
            }
            _send(self, 200, json.dumps(manifest).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/log":
            _send(self, 200, b"hello\n", "text/plain; charset=utf-8")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def test_list_artifact_files_removed():
    # list_artifact_files は削除された（仕様変更）
    assert not hasattr(vt, "list_artifact_files")


def test_get_artifact_file(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        out_path = tmp_path / "solver.log"
        data = vt.get_artifact_file(
            f"http://127.0.0.1:{port}",
            ARTIFACT_DIR,
            "solver.log",
            output_path=str(out_path),
        )
        assert data == b"hello\n"
        assert out_path.read_bytes() == b"hello\n"
    finally:
        server.shutdown()
        server.server_close()


class _HandlerDF(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == f"/artifacts/{ARTIFACT_DIR}/manifest":
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

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/schema":
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

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/vent_flow_rate":
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def test_get_artifact_file_df_sets_time_index_from_manifest_output_index():
    server = HTTPServer(("127.0.0.1", 0), _HandlerDF)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        df = vt.get_artifact_file(
            base_url,
            ARTIFACT_DIR,
            "vent.flow_rate.f32.bin",
            output_path=None,
        )
        assert list(df.columns) == ["c1", "c2"]
        assert df.shape == (2, 2)
        assert str(df.index[0]) == "2025-01-01 00:00:00"
        assert str(df.index[1]) == "2025-01-01 01:00:00"
        assert df.index.name == "time"
    finally:
        server.shutdown()
        server.server_close()

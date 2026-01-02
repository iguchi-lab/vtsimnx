import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

import vtsimnx as vt


def _serve(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


class _BaseDFHandler(BaseHTTPRequestHandler):
    schema_dtype = "f32le"
    schema_layout = "timestep-major"
    schema_length = 2
    index_length = 2
    bin_size_ok = True

    def do_GET(self):
        if self.path == "/work/output.artifacts.123/schema.json":
            schema = {
                "dtype": self.schema_dtype,
                "layout": self.schema_layout,
                "length": self.schema_length,
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

        if self.path == "/work/output.artifacts.123/manifest.json":
            manifest = {
                "output": {
                    "index": {
                        "start": "2025-01-01 00:00:00",
                        "end": "2025-01-01 01:00:00",
                        "timestep": 3600,
                        "length": self.index_length,
                    },
                    "result_files": {
                        "vent_flow_rate": "vent.flow_rate.f32.bin",
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

        if self.path == "/work/output.artifacts.123/vent.flow_rate.f32.bin":
            if self.bin_size_ok:
                arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            else:
                arr = np.array([1.0, 2.0, 3.0], dtype=np.dtype("<f4"))  # mismatch
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


class _BadDtypeHandler(_BaseDFHandler):
    schema_dtype = "f64le"


def test_get_artifact_file_raises_on_unexpected_dtype():
    server, port = _serve(_BadDtypeHandler)
    try:
        with pytest.raises(ValueError, match="dtype"):
            _ = vt.get_artifact_file(
                f"http://127.0.0.1:{port}",
                "output.artifacts.123",
                "vent.flow_rate.f32.bin",
                output_path=None,
            )
    finally:
        server.shutdown()
        server.server_close()


class _BadLayoutHandler(_BaseDFHandler):
    schema_layout = "column-major"


def test_get_artifact_file_raises_on_unexpected_layout():
    server, port = _serve(_BadLayoutHandler)
    try:
        with pytest.raises(ValueError, match="layout"):
            _ = vt.get_artifact_file(
                f"http://127.0.0.1:{port}",
                "output.artifacts.123",
                "vent.flow_rate.f32.bin",
                output_path=None,
            )
    finally:
        server.shutdown()
        server.server_close()


class _BinSizeMismatchHandler(_BaseDFHandler):
    bin_size_ok = False


def test_get_artifact_file_raises_on_bin_size_mismatch():
    server, port = _serve(_BinSizeMismatchHandler)
    try:
        with pytest.raises(ValueError, match="要素数が不一致"):
            _ = vt.get_artifact_file(
                f"http://127.0.0.1:{port}",
                "output.artifacts.123",
                "vent.flow_rate.f32.bin",
                output_path=None,
            )
    finally:
        server.shutdown()
        server.server_close()


class _IndexLengthMismatchHandler(_BaseDFHandler):
    index_length = 999


def test_get_artifact_file_does_not_set_time_index_when_index_length_mismatch():
    server, port = _serve(_IndexLengthMismatchHandler)
    try:
        df = vt.get_artifact_file(
            f"http://127.0.0.1:{port}",
            "output.artifacts.123",
            "vent.flow_rate.f32.bin",
            output_path=None,
        )
        assert df is not None
        # mismatch の場合は index は付与されない（RangeIndexのまま）
        assert df.index.name != "time"
    finally:
        server.shutdown()
        server.server_close()



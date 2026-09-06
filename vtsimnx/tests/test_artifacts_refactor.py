"""artifacts リファクタ（高・中優先度）の回帰テスト。"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from vtsimnx.artifacts import (
    ArtifactClient,
    ArtifactDecodeError,
    ArtifactNotFound,
    decode_f32_series,
)
from vtsimnx.artifacts._schema import NormalizedManifest
from vtsimnx.units import VOLUME_FLOW_M3_S

ARTIFACT_DIR = "output.artifacts.refactor"


def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _CountingHandler(BaseHTTPRequestHandler):
    counts: dict[str, int] = {}

    def do_GET(self):
        key = self.path
        _CountingHandler.counts[key] = _CountingHandler.counts.get(key, 0) + 1

        if self.path == f"/artifacts/{ARTIFACT_DIR}/manifest":
            manifest = {
                "output": {
                    "index": {
                        "start": "2025-01-01 00:00:00",
                        "end": "2025-01-01 01:00:00",
                        "timestep": 3600,
                        "length": 2,
                    },
                    "log_file": "solver.log",
                    "result_files": {
                        "schema": "schema.json",
                        "vent_flow_rate": "vent.flow_rate.f32.bin",
                    },
                },
                "files": {
                    "schema": "schema.json",
                    "log": "solver.log",
                },
            }
            _send(self, 200, json.dumps(manifest).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/schema":
            schema = {
                "dtype": "f32le",
                "layout": "timestep-major",
                "length": 2,
                "series": {"vent_flow_rate": {"keys": ["c1", "c2"]}},
            }
            _send(self, 200, json.dumps(schema).encode("utf-8"), "application/json; charset=utf-8")
            return

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/vent_flow_rate":
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))
            _send(self, 200, arr.tobytes(), "application/octet-stream")
            return

        if self.path == f"/artifacts/{ARTIFACT_DIR}/download/log":
            _send(self, 200, b"log-ok\n", "text/plain; charset=utf-8")
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


@pytest.fixture
def artifact_server():
    _CountingHandler.counts = {}
    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def test_normalized_manifest_resolves_filename_and_key():
    nm = NormalizedManifest.from_dict(
        {
            "output": {
                "log_file": "solver.log",
                "result_files": {"vent_flow_rate": "vent.flow_rate.f32.bin", "schema": "schema.json"},
            }
        }
    )
    assert nm.resolve_download_key("vent.flow_rate.f32.bin") == "vent_flow_rate"
    assert nm.resolve_download_key("vent_flow_rate") == "vent_flow_rate"
    assert nm.resolve_download_key("solver.log") == "log"
    with pytest.raises(ArtifactNotFound):
        nm.resolve_download_key("missing.bin")


def test_decode_f32_series_attaches_units():
    schema = {
        "dtype": "f32le",
        "layout": "timestep-major",
        "length": 2,
        "series": {"vent_flow_rate": {"keys": ["a", "b"]}},
    }
    raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4")).tobytes()
    df = decode_f32_series(raw, schema, "vent_flow_rate")
    assert df.attrs["unit"] == VOLUME_FLOW_M3_S
    assert df.attrs["series"] == "vent_flow_rate"


def test_decode_f32_series_empty_keys_is_zero_columns():
    """エンジンは対象なし系列を keys=[]・bin 0 バイトで出す。1 列扱いにしない。"""
    schema = {
        "dtype": "f32le",
        "layout": "timestep-major",
        "length": 3,
        "series": {"humidity_x": {"keys": []}},
    }
    df = decode_f32_series(
        b"",
        schema,
        "humidity_x",
        index_spec={
            "start": "2025-01-01 00:00:00",
            "timestep": 3600,
            "length": 3,
        },
    )
    assert list(df.columns) == []
    assert df.shape == (3, 0)
    assert df.index.name == "time"


def test_decode_f32_series_raises_artifact_decode_error():
    schema = {
        "dtype": "f32le",
        "layout": "timestep-major",
        "length": 2,
        "series": {"vent_flow_rate": {"keys": ["a"]}},
    }
    with pytest.raises(ArtifactDecodeError):
        decode_f32_series(b"\x00\x00", schema, "vent_flow_rate")
    with pytest.raises(ValueError):
        decode_f32_series(b"\x00\x00", schema, "vent_flow_rate")


def test_artifact_client_caches_manifest_and_schema(artifact_server):
    client = ArtifactClient(artifact_server, ARTIFACT_DIR)
    df1 = client.get_series_df("vent_flow_rate")
    df2 = client.get_series_df("vent_flow_rate")
    assert df1.shape == (2, 2)
    assert df2.shape == (2, 2)
    assert df1.attrs["unit"] == VOLUME_FLOW_M3_S

    manifest_path = f"/artifacts/{ARTIFACT_DIR}/manifest"
    schema_path = f"/artifacts/{ARTIFACT_DIR}/download/schema"
    assert _CountingHandler.counts[manifest_path] == 1
    assert _CountingHandler.counts[schema_path] == 1
    assert _CountingHandler.counts[f"/artifacts/{ARTIFACT_DIR}/download/vent_flow_rate"] == 2


def test_get_artifact_file_sets_unit_attrs(artifact_server):
    import vtsimnx as vt

    df = vt.get_artifact_file(
        artifact_server,
        ARTIFACT_DIR,
        "vent.flow_rate.f32.bin",
        output_path=None,
    )
    assert df.attrs["unit"] == VOLUME_FLOW_M3_S
    assert df.attrs["series"] == "vent_flow_rate"


def test_get_artifact_bytes_stable_export(artifact_server):
    import vtsimnx as vt

    data = vt.get_artifact_bytes(artifact_server, ARTIFACT_DIR, "solver.log")
    assert data == b"log-ok\n"

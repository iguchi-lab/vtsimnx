import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import vtsimnx as vt


class _State:
    content_encoding = None
    received = None


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        _State.content_encoding = self.headers.get("Content-Encoding")
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n > 0 else b""

        if _State.content_encoding == "gzip":
            body = gzip.decompress(body)

        _State.received = json.loads(body.decode("utf-8")) if body else None

        resp = {"result": {"artifact_dir": "output.artifacts.1", "result_files": {}}}
        out = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, format, *args):
        return


def test_run_calc_compress_request_gzip():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        cfg = {"simulation": {"index": {"length": 2, "timestep": 1}}}
        _ = vt.run_calc(base_url, cfg, output_path=None, with_dataframes=False, compress_request=True)

        assert _State.content_encoding == "gzip"
        assert isinstance(_State.received, dict)
        assert _State.received["config"] == cfg
    finally:
        server.shutdown()
        server.server_close()



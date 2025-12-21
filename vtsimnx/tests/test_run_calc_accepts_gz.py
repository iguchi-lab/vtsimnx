import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def test_run_calc_accepts_json_gz(tmp_path):
    cfg = {"simulation": {"index": {"length": 2, "timestep": 1}}}
    p = tmp_path / "cfg.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(cfg, f)

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        _ = vt.run_calc(base_url, p, output_path=None, with_dataframes=False)

        assert isinstance(_State.received, dict)
        assert _State.received["config"] == cfg
    finally:
        server.shutdown()
        server.server_close()



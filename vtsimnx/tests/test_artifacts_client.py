import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import vtsimnx as vt


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # 期待: /artifacts/<artifact_dir>/files
        if self.path == "/artifacts/output.artifacts.123/files":
            body = {
                "artifact_dir": "output.artifacts.123",
                "files": [
                    {"name": "solver.log", "size_bytes": 10, "url": "/artifacts/output.artifacts.123/files/solver.log"},
                    {"name": "schema.json", "size_bytes": 20, "url": "/artifacts/output.artifacts.123/files/schema.json"},
                ],
            }
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # テスト出力を静かにする
        return


def test_list_artifact_files(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        out_path = tmp_path / "artifact_files.json"
        data = vt.list_artifact_files(
            f"http://127.0.0.1:{port}",
            "output.artifacts.123",
            output_path=str(out_path),
        )
        assert data["artifact_dir"] == "output.artifacts.123"
        assert len(data["files"]) == 2
        assert out_path.exists()
    finally:
        server.shutdown()
        server.server_close()



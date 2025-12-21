import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import vtsimnx as vt


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # get_artifact_file の想定: /work/<artifact_dir>/<filename>
        if self.path in (
            "/work/output.artifacts.123/solver.log",
            "/work/output.artifacts.123/artifacts/solver.log",  # フォールバック経路
        ):
            raw = b"hello\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # テスト出力を静かにする
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
            "output.artifacts.123",
            "solver.log",
            output_path=str(out_path),
        )
        assert data == b"hello\n"
        assert out_path.read_bytes() == b"hello\n"
    finally:
        server.shutdown()
        server.server_close()



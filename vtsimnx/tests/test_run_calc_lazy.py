import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import vtsimnx as vt


class _State:
    post_run = 0
    get_work = 0
    get_schema = 0
    get_manifest = 0
    get_bin = 0
    get_log = 0


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        _State.post_run += 1

        # Windows環境だと、POSTボディを読まずに応答すると接続中断扱いになることがある。
        # ここで Content-Length 分を読み捨てておく。
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n > 0:
            _ = self.rfile.read(n)

        # 結果（/run レスポンス）: log.text があるので log 取得はHTTP不要にできる
        body = {
            "result": {
                "artifact_dir": "output.artifacts.123",
                "log_file": "solver.log",
                "log": {"text": "preloaded log"},
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
        # get_artifact_file は /work/{artifact_dir}/{filename} を叩く想定
        if self.path.startswith("/work/output.artifacts.123/"):
            _State.get_work += 1

        if self.path == "/work/output.artifacts.123/schema.json":
            _State.get_schema += 1
            schema = {
                "dtype": "f32le",
                "layout": "timestep-major",
                "length": 2,
                "series": {"vent_flow_rate": {"keys": ["c1", "c2"]}},
            }
            raw = json.dumps(schema).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path == "/work/output.artifacts.123/manifest.json":
            _State.get_manifest += 1
            manifest = {
                "result": {
                    "result_files": {"vent_flow_rate": "vent.flow_rate.f32.bin"},
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
            _State.get_bin += 1
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.dtype("<f4"))  # (T=2, N=2)
            raw = arr.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if self.path == "/work/output.artifacts.123/solver.log":
            _State.get_log += 1
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
        return


def test_run_calc_with_dataframes_is_lazy():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        res = vt.run_calc(base_url, {"simulation": {"index": {"length": 2, "timestep": 1}}}, output_path=None, with_dataframes=True)

        # /run 以外の GET は、まだ走っていない（遅延ロード）
        assert _State.post_run == 1
        assert _State.get_work == 0
        assert hasattr(res, "get_series_df")

        # log はレスポンス内に埋まっているので GET なし
        assert res.log == "preloaded log"
        assert _State.get_log == 0

        # DataFrame を要求したときだけ GET が走る
        df = res.get_series_df("vent_flow_rate")
        assert df is not None
        assert list(df.columns) == ["c1", "c2"]
        assert df.shape == (2, 2)
        assert _State.get_schema >= 1
        assert _State.get_manifest >= 1
        assert _State.get_bin >= 1
    finally:
        server.shutdown()
        server.server_close()



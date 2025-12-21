import gzip
import json

import vtsimnx as vt


def test_read_json_gz(tmp_path):
    p = tmp_path / "x.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"a": 1, "b": [1, 2, 3]}, f)

    obj = vt.read_json(p)
    assert obj["a"] == 1
    assert obj["b"] == [1, 2, 3]



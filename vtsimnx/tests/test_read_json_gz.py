import gzip
import json

import pytest

import vtsimnx as vt


def test_read_json_gz(tmp_path):
    p = tmp_path / "x.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump({"a": 1, "b": [1, 2, 3]}, f)

    obj = vt.read_json(p)
    assert obj["a"] == 1
    assert obj["b"] == [1, 2, 3]


def test_read_json_invalid_json_raises_value_error(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError):
        _ = vt.read_json(p)


def test_read_json_missing_file_raises_file_not_found(tmp_path):
    p = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        _ = vt.read_json(p)



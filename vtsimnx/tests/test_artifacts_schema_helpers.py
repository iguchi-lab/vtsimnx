import pytest

from vtsimnx.artifacts._schema import extract_result_files


def test_extract_result_files_accepts_output_result_files():
    manifest = {"output": {"result_files": {"a": "a.bin"}}}
    assert extract_result_files(manifest) == {"a": "a.bin"}


def test_extract_result_files_accepts_result_result_files():
    manifest = {"result": {"result_files": {"a": "a.bin"}}}
    assert extract_result_files(manifest) == {"a": "a.bin"}


def test_extract_result_files_accepts_top_level_result_files():
    manifest = {"result_files": {"a": "a.bin"}}
    assert extract_result_files(manifest) == {"a": "a.bin"}


def test_extract_result_files_accepts_files_alias():
    manifest = {"files": {"a": "a.bin"}}
    assert extract_result_files(manifest) == {"a": "a.bin"}


def test_extract_result_files_raises_when_missing():
    with pytest.raises(ValueError):
        extract_result_files({})



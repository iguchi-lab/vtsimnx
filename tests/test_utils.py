import numpy as np
import pandas as pd
import pytest
import vtsimnx.utils as utils


# _expand_chain --------------------------------------------------------------
def test_expand_chain_basic():
    # 連鎖 "A->B->C" がセグメント ["A->B", "B->C"] に展開される
    assert utils._expand_chain("A->B->C") == ["A->B", "B->C"]


def test_expand_chain_trims_and_void_padding():
    # 前後のスペースは削除され、空要素は "void" でパディングされる
    assert utils._expand_chain("  A  ->  B  ") == ["A->B"]
    assert utils._expand_chain("->B->") == ["void->B", "B->void"]


def test_expand_chain_too_short_raises():
    # ノード1個以下はエラー
    with pytest.raises(ValueError):
        utils._expand_chain("A")


# _split_key_and_comment -----------------------------------------------------
def test_split_key_and_comment_basic():
    # "key||comment" を分割し、空白はトリムされる
    key, c = utils._split_key_and_comment(" AAA || BBB ")
    assert key == "AAA" and c == "BBB"


def test_split_key_and_comment_no_comment():
    # コメントが無ければ、キーのみを返す
    key, c = utils._split_key_and_comment("AAA")
    assert key == "AAA" and c == ""


# _split_compound_key --------------------------------------------------------
def test_split_compound_key_basic():
    # AND 条件の複合キー "A&&B&&C" を分割
    parts = utils._split_compound_key(" A && B && C ")
    assert parts == ["A", "B", "C"]


def test_split_compound_key_single():
    assert utils._split_compound_key("A") == ["A"]


# _normalize_timeseries_mapping / _append_with_comment ----------------------
def test_normalize_timeseries_mapping_pd_series_and_numpy_array():
    # pandas.Series / numpy.ndarray は list に正規化される
    d = {
        "s": pd.Series([1, 2, 3]),
        "a": np.array([4, 5]),
        "x": 10,
    }
    out = utils._normalize_timeseries_mapping(d)
    assert out["s"] == [1, 2, 3]
    assert out["a"] == [4, 5]
    assert out["x"] == 10


def test_append_with_comment_merges_and_normalizes():
    base = {"a": 1, "b": np.array([1, 2]), "c": pd.Series([3, 4])}
    # 上書き時に正規化が走ること
    out = utils._append_with_comment(base, b=pd.Series([9]), d="x")
    assert out["a"] == 1
    assert out["b"] == [9]
    assert out["c"] == [3, 4]
    assert out["d"] == "x"


# convert_numeric_values -----------------------------------------------------
def test_convert_numeric_values_array_and_bool_keys():
    # 文字列数値は数値化、数値配列は numpy.ndarray へ、bool_keys は True/False に
    data = {"arr": ["1", "2.5", "3e0"], "flag": "1"}
    out = utils.convert_numeric_values(data, bool_keys={"flag"})
    assert isinstance(out["arr"], np.ndarray)
    assert out["arr"].dtype.kind in ("i", "f")
    assert out["flag"] is True


def test_convert_numeric_values_nested_and_non_numeric_string():
    data = {
        "nested": ["1", "2", "x"],   # 数値と非数値が混在 → Python リストのまま
        "empty": [],
        "num_list": ["1", "2", "3"], # すべて数値 → numpy.ndarray
    }
    out = utils.convert_numeric_values(data)
    # 混在した場合でも、数値に変換可能な要素は数値化される（文字列のままにはならない）
    assert out["nested"] == [1, 2, "x"]
    assert out["empty"] == []
    assert isinstance(out["num_list"], np.ndarray)
    assert out["num_list"].tolist() == [1, 2, 3]


# convert_to_json_compatible -------------------------------------------------
def test_convert_to_json_compatible_basic():
    data = {"a": np.array([1, 2]), "b": np.float64(1.5), "c": np.bool_(True)}
    out = utils.convert_to_json_compatible(data)
    assert out == {"a": [1, 2], "b": 1.5, "c": True}


def test_convert_to_json_compatible_nested_list_and_scalars():
    data = {
        "list": [np.array([1, 2]), np.bool_(False), 3],
        "scalar_i": np.int64(10),
        "scalar_f": np.float32(2.5),
    }
    out = utils.convert_to_json_compatible(data)
    assert out["list"][0] == [1, 2]
    assert out["list"][1] is False
    assert out["scalar_i"] == 10
    assert out["scalar_f"] == 2.5


# ensure_timeseries ----------------------------------------------------------
def test_ensure_timeseries_from_list_and_array_and_scalar():
    assert utils.ensure_timeseries([1, 2, 3], length=5) == [1, 2, 3]
    assert utils.ensure_timeseries(np.array([4, 5]), length=5) == [4, 5]
    # スカラーは length 個に複製される
    assert utils.ensure_timeseries(7, length=3) == [7, 7, 7]

import numpy as np
import vtsimnx.utils as utils


def test_expand_chain():
    assert utils._expand_chain("A->B->C") == ["A->B", "B->C"]


def test_convert_numeric_values_array():
    data = {"arr": ["1", "2.5", "3e0"], "flag": "1"}
    out = utils.convert_numeric_values(data, bool_keys={"flag"})
    assert isinstance(out["arr"], np.ndarray)
    assert out["arr"].dtype.kind in ("i", "f")
    assert out["flag"] is True


def test_convert_to_json_compatible():
    data = {"a": np.array([1, 2]), "b": np.float64(1.5), "c": np.bool_(True)}
    out = utils.convert_to_json_compatible(data)
    assert out == {"a": [1, 2], "b": 1.5, "c": True}



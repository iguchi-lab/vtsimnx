import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from vtsimnx.utils.jsonable import to_jsonable


def test_to_jsonable_handles_pandas_numpy_and_specials():
    s = pd.Series([1.0, np.nan, np.inf, -np.inf, 2.0])
    df = pd.DataFrame({"a": [1, 2], "b": [np.int64(3), np.float64(4.0)]})
    obj = {
        "series": s,
        "df": df,
        "dt": datetime(2026, 1, 2, 3, 4, 5),
        "path": Path("x/y/z.txt"),
        "arr": np.array([1, 2, 3]),
        "scalar_i": np.int64(7),
        "scalar_f": np.float64(8.5),
    }

    out = to_jsonable(obj)

    assert isinstance(out, dict)
    assert out["series"] == [1.0, None, None, None, 2.0]
    assert out["df"] == {"a": [1, 2], "b": [3, 4.0]}
    assert out["dt"] == "2026-01-02T03:04:05"
    assert out["path"] == str(Path("x/y/z.txt"))
    assert out["arr"] == [1, 2, 3]
    assert out["scalar_i"] == 7
    assert out["scalar_f"] == 8.5

    # 変換後はJSON互換（NaN/Inf を含まない）になっていること
    assert all(not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in out["series"] if v is not None)



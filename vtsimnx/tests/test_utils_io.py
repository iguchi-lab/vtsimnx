import numpy as np
import pandas as pd
import pytest

import vtsimnx as vt


def _write_dummy_hasp(path, *, last_hour_overrides=None):
    """365日×7行の最小HASPデータを作成する。"""
    base = [500, 50, 100, 200, 300, 8, 15]  # t, h, i_b, i_d, n_r, w_d, w_s
    overrides = last_hour_overrides or {}

    with open(path, "wb") as f:
        for day in range(365):
            for ch in range(7):
                vals = [base[ch]] * 24
                if day == 364 and ch in overrides:
                    vals[-1] = int(overrides[ch])
                line = "".join(f"{v:03d}" for v in vals) + "\n"
                f.write(line.encode("ascii"))


def test_read_hasp_valid_and_rotates_last_hour_to_head(tmp_path):
    p = tmp_path / "dummy.has"
    # 最終時刻(12/31 24:00相当)の外気温 raw=650 -> 15.0[℃]
    _write_dummy_hasp(p, last_hour_overrides={0: 650})

    df = vt.read_hasp(p)

    assert len(df) == 365 * 24
    assert df.index[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert list(df.columns) == [
        "外気温",
        "外気絶対湿度",
        "直達日射量",
        "水平面拡散日射量",
        "夜間放射量",
        "風向",
        "風速",
    ]
    assert np.isclose(float(df.iloc[0]["外気温"]), 15.0)
    assert np.isclose(float(df.iloc[1]["外気温"]), 0.0)
    assert np.isclose(float(df.iloc[1]["外気絶対湿度"]), 5.0)
    assert np.isclose(float(df.iloc[1]["直達日射量"]), 116.222, atol=1e-6)
    assert int(df.iloc[1]["風向"]) == 8
    assert np.isclose(float(df.iloc[1]["風速"]), 1.5)


def test_read_hasp_raises_on_too_few_lines(tmp_path):
    p = tmp_path / "short.has"
    p.write_text("500" * 24 + "\n", encoding="ascii")

    with pytest.raises(ValueError):
        _ = vt.read_hasp(p)


def test_read_hasp_raises_on_non_ascii(tmp_path):
    p = tmp_path / "non_ascii.has"
    with open(p, "wb") as f:
        f.write(b"\xff\xfe\xfd")

    with pytest.raises(ValueError):
        _ = vt.read_hasp(p)


def test_read_csv_bfill_then_ffill(tmp_path):
    p = tmp_path / "x.csv"
    idx = pd.date_range("2026-01-01 00:00:00", periods=5, freq="1h")
    src = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0, np.nan]}, index=idx)
    src.to_csv(p)

    out = vt.read_csv(p)
    assert out.index.equals(idx)
    assert out["a"].to_list() == [1.0, 1.0, 3.0, 3.0, 3.0]

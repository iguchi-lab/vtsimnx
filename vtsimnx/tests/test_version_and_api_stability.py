"""バージョン正本と公開 API 安定性のテスト。"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def _pyproject_version() -> str:
    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("version") and "=" in s:
                return s.split("=", 1)[1].strip().strip('"').strip("'")
        raise AssertionError("version not found in pyproject.toml")
    data = tomllib.loads(text)
    return str(data["project"]["version"])


def test_get_version_matches_pyproject():
    from vtsimnx import __version__, get_version

    expected = _pyproject_version()
    assert get_version() == expected
    assert __version__ == expected


def test_units_field_and_series():
    from vtsimnx.units import FIELD_UNITS, SERIES_UNITS, unit_for_field, unit_for_series

    assert unit_for_field("t") == "degC"
    assert unit_for_field("p") == "Pa"
    assert unit_for_field("vol") == "m3/s"
    assert unit_for_series("vent_flow_rate") == "m3/s"
    assert "heat_generation" in FIELD_UNITS
    assert "thermal_temperature" in SERIES_UNITS


def test_api_stability_catalog():
    from vtsimnx.api_stability import DEPRECATED, EXPERIMENTAL, STABLE

    stable_names = {s.name for s in STABLE}
    assert "run_calc" in stable_names
    assert "get_version" in stable_names
    assert any(s.name == "make_8760_data" for s in DEPRECATED)
    assert any(s.name == "solar_gain_by_angles" for s in EXPERIMENTAL)


def test_deprecated_top_level_warns():
    import vtsimnx

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = vtsimnx.make_8760_data
    assert fn is not None
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

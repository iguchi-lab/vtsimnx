"""パッケージ版の参照（正本はリポジトリ直下 pyproject.toml）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _read_pyproject_version() -> str | None:
    # engine/app/versioning.py -> engine/app -> engine -> repo
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("version") and "=" in s:
                return s.split("=", 1)[1].strip().strip('"').strip("'")
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ver = data.get("project", {}).get("version")
    return str(ver) if ver else None


@lru_cache(maxsize=1)
def get_package_version() -> str:
    """client パッケージ版 = API 版（monorepo 同一リリース）。"""
    src = _read_pyproject_version()
    if src:
        return src
    try:
        from vtsimnx import get_version

        return get_version()
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("vtsimnx")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    return "0.0.0"


# solver output JSON の format_version（C++ 側と同期）
SCHEMA_FORMAT_VERSION = 5

__all__ = ["get_package_version", "SCHEMA_FORMAT_VERSION"]

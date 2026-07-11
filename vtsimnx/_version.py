"""パッケージ版の正本参照。

正本はリポジトリ直下 `pyproject.toml` の `[project].version`。
ソースツリー上では常に pyproject を優先し、wheel インストール環境
（隣に pyproject が無い場合）では `importlib.metadata` を使う。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _read_pyproject_version() -> str | None:
    # vtsimnx/_version.py -> vtsimnx/ -> repo root（ソース配置時）
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("version") and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
            return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ver = data.get("project", {}).get("version")
    return str(ver) if ver else None


def _read_metadata_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return None
    try:
        return version("vtsimnx")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """pyproject（ソース）があればそれを、なければインストール版メタデータを返す。"""
    src = _read_pyproject_version()
    if src:
        return src
    meta = _read_metadata_version()
    if meta:
        return meta
    return "0.0.0"


__version__ = get_version()

__all__ = ["__version__", "get_version"]

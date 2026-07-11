"""artifacts クライアント用例外。"""
from __future__ import annotations


class ArtifactError(Exception):
    """成果物取得・復元の基底例外。"""


class ArtifactNotFound(KeyError, ArtifactError):
    """manifest にキー/ファイル名が無い、または series が見つからない。"""


class ArtifactDecodeError(ValueError, ArtifactError):
    """schema / f32.bin の形式・サイズ不一致など復元失敗。"""


class ArtifactHTTPError(RuntimeError, ArtifactError):
    """成果物 API の HTTP 失敗。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


__all__ = [
    "ArtifactError",
    "ArtifactNotFound",
    "ArtifactDecodeError",
    "ArtifactHTTPError",
]

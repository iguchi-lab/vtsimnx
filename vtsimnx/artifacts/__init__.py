"""成果物クライアント（HTTP 取得・f32 復元）。

公開:
  - get_artifact_file / get_artifact_bytes（stable）
  - ArtifactClient（manifest/schema キャッシュ）
  - decode_f32_series（ローカル/ツール共用）
  - ArtifactNotFound / ArtifactDecodeError / ArtifactHTTPError
"""
from .client import ArtifactClient
from .errors import ArtifactDecodeError, ArtifactError, ArtifactHTTPError, ArtifactNotFound
from .get_artifact_file import get_artifact_bytes, get_artifact_file
from ._decode import decode_f32_series
from ._schema import NormalizedManifest, extract_manifest_error, extract_result_files, series_columns

__all__ = [
    "ArtifactClient",
    "ArtifactError",
    "ArtifactNotFound",
    "ArtifactDecodeError",
    "ArtifactHTTPError",
    "NormalizedManifest",
    "decode_f32_series",
    "extract_manifest_error",
    "extract_result_files",
    "get_artifact_bytes",
    "get_artifact_file",
    "series_columns",
]

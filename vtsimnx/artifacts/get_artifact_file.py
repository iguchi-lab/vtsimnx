"""成果物取得の公開関数（後方互換ラッパ）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from .client import ArtifactClient


def get_artifact_bytes(
    base_url: str,
    artifact_dir: str,
    filename: str,
    *,
    output_path: Optional[str] = None,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    成果物ディレクトリからファイルを1つ取得して bytes を返す（復元はしない）。

    想定API:
      GET {base_url}/artifacts/{artifact_dir}/manifest
      GET {base_url}/artifacts/{artifact_dir}/download/{key}

    - filename は実ファイル名（例: schema.json）または download key（例: schema）
    - output_path を指定すると保存も行う
    - manifest を渡すと再 GET を省略できる
    """
    client = ArtifactClient(base_url, artifact_dir, api_key=api_key, timeout=timeout)
    return client.get_bytes(filename, output_path=output_path, manifest=manifest)


def get_artifact_file(
    base_url: str,
    artifact_dir: str,
    filename: str,
    output_path: Optional[str] = None,
    *,
    index_spec: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> Union[bytes, "pd.DataFrame"]:
    """
    成果物ディレクトリからファイルを1つ取得する。

    想定API:
      GET {base_url}/artifacts/{artifact_dir}/manifest
      GET {base_url}/artifacts/{artifact_dir}/download/{key}

    - output_path を指定すると、取得した内容をそのパスに保存する（Noneなら保存しない）
    - `.f32.bin` の場合は `schema` と `manifest` を参照して DataFrame に復元して返す
      - dtype: schema.json の "f32le" -> np.dtype("<f4")
      - layout: schema.json の "timestep-major" -> shape=(T, N)
      - 単位は ``DataFrame.attrs["unit"]``（``vtsimnx.units.SERIES_UNITS``）
    - それ以外は取得したバイト列を返す
    """
    client = ArtifactClient(base_url, artifact_dir, api_key=api_key, timeout=timeout)
    return client.get_file(filename, output_path=output_path, index_spec=index_spec)


__all__ = ["get_artifact_bytes", "get_artifact_file"]

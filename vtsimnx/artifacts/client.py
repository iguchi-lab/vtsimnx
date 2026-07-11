"""成果物取得クライアント（manifest / schema キャッシュ付き）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from ._decode import decode_f32_series, load_json_bytes
from ._http import download_by_key, fetch_normalized_manifest
from ._schema import NormalizedManifest
from .errors import ArtifactNotFound


class ArtifactClient:
    """
    同一 artifact_dir に対する取得をまとめる。

    - manifest / schema をキャッシュし、系列を複数取るときの再 GET を減らす
    - 公開関数 get_artifact_file / get_artifact_bytes の実装基盤
    """

    def __init__(
        self,
        base_url: str,
        artifact_dir: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.artifact_dir = artifact_dir
        self.api_key = api_key
        self.timeout = float(timeout)
        self._manifest: Optional[NormalizedManifest] = None
        self._schema: Optional[Dict[str, Any]] = None

    def clear_cache(self) -> None:
        self._manifest = None
        self._schema = None

    def seed_manifest(self, manifest: Union[Dict[str, Any], NormalizedManifest]) -> NormalizedManifest:
        """
        既知の manifest / run レスポンスでキャッシュを埋める（追加の manifest GET を避ける）。
        """
        if isinstance(manifest, NormalizedManifest):
            self._manifest = manifest
        else:
            self._manifest = NormalizedManifest.from_dict(manifest)
        return self._manifest

    def get_manifest(self, *, refresh: bool = False) -> NormalizedManifest:
        if self._manifest is None or refresh:
            self._manifest = fetch_normalized_manifest(
                self.base_url,
                self.artifact_dir,
                timeout=self.timeout,
                api_key=self.api_key,
            )
        return self._manifest

    def get_schema(self, *, refresh: bool = False) -> Dict[str, Any]:
        if self._schema is None or refresh:
            raw = self.get_bytes("schema", use_cached_manifest=True)
            self._schema = load_json_bytes(raw)
        return self._schema

    def get_bytes(
        self,
        name_or_filename: str,
        *,
        output_path: Optional[str] = None,
        use_cached_manifest: bool = True,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        if manifest is not None:
            nm = NormalizedManifest.from_dict(manifest)
        elif use_cached_manifest:
            nm = self.get_manifest()
        else:
            nm = self.get_manifest(refresh=True)
        key = nm.resolve_download_key(name_or_filename)
        data = download_by_key(
            self.base_url,
            self.artifact_dir,
            key,
            timeout=self.timeout,
            api_key=self.api_key,
        )
        if output_path is not None:
            with open(output_path, "wb") as f:
                f.write(data)
        return data

    def get_file(
        self,
        name_or_filename: str,
        output_path: Optional[str] = None,
        *,
        index_spec: Optional[Dict[str, Any]] = None,
    ) -> Union[bytes, pd.DataFrame]:
        nm = self.get_manifest()
        key = nm.resolve_download_key(name_or_filename)
        data = download_by_key(
            self.base_url,
            self.artifact_dir,
            key,
            timeout=self.timeout,
            api_key=self.api_key,
        )
        if output_path is not None:
            with open(output_path, "wb") as f:
                f.write(data)

        result_files = nm.result_files
        bin_basename = result_files.get(key) if key in result_files else name_or_filename.split("/")[-1]
        if not isinstance(bin_basename, str):
            bin_basename = name_or_filename.split("/")[-1]

        if not bin_basename.endswith(".f32.bin"):
            return data

        series_name: Optional[str] = None
        for k, v in result_files.items():
            if v == bin_basename:
                series_name = k
                break
        if series_name is None and key in result_files:
            series_name = key
        if series_name is None:
            raise ArtifactNotFound(f"manifest.json から {bin_basename} に対応する series 名が見つかりませんでした")

        schema = self.get_schema()
        if index_spec is None:
            index_spec = nm.index
        return decode_f32_series(
            data,
            schema,
            series_name,
            index_spec=index_spec,
            source_name=bin_basename,
        )

    def get_series_df(
        self,
        series_name: str,
        *,
        index_spec: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        nm = self.get_manifest()
        fname = nm.result_files.get(series_name)
        if not isinstance(fname, str) or not fname:
            raise ArtifactNotFound(f"series not in result_files: {series_name!r}")
        if not fname.endswith(".f32.bin"):
            raise ArtifactNotFound(f"series is not f32.bin: {series_name!r} -> {fname!r}")
        data = self.get_bytes(fname)
        schema = self.get_schema()
        if index_spec is None:
            index_spec = nm.index
        return decode_f32_series(
            data,
            schema,
            series_name,
            index_spec=index_spec,
            source_name=fname,
        )


__all__ = ["ArtifactClient"]

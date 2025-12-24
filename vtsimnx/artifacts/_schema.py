from __future__ import annotations

from typing import Any, Dict, List


def extract_result_files(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    manifest/manifest相当のJSONから「系列名 -> ファイル名」の辞書を取り出す。

    想定される形（いずれか）:
      - {"output": {"result_files": {...}}}
      - {"result": {"result_files": {...}}}
      - {"result_files": {...}}
      - {"files": {...}}
    """
    if isinstance(manifest.get("output"), dict) and isinstance(manifest["output"].get("result_files"), dict):
        result_files = manifest["output"]["result_files"]
    elif isinstance(manifest.get("result"), dict) and isinstance(manifest["result"].get("result_files"), dict):
        result_files = manifest["result"]["result_files"]
    elif isinstance(manifest.get("result_files"), dict):
        result_files = manifest["result_files"]
    elif isinstance(manifest.get("files"), dict):
        result_files = manifest["files"]
    else:
        raise ValueError("manifest.json から result_files/files が見つかりませんでした")

    out: Dict[str, str] = {}
    for k, v in result_files.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def series_columns(schema: Dict[str, Any], series_name: str) -> List[str]:
    """
    schema.json から指定 series の列名配列を取り出す。

    - series.<name>.keys が [] の場合はスカラー扱いで [series_name] を返す
    """
    series = schema.get("series")
    if not isinstance(series, dict) or series_name not in series:
        raise KeyError(f"schema.json に series.{series_name} がありません")

    spec = series[series_name]
    if not isinstance(spec, dict):
        raise TypeError(f"schema.json の series.{series_name} が不正です")

    keys = spec.get("keys", [])
    if keys is None:
        keys = []

    if not isinstance(keys, list):
        raise TypeError(f"schema.json の series.{series_name}.keys が配列ではありません")

    if len(keys) == 0:
        return [series_name]

    cols: List[str] = []
    for k in keys:
        if not isinstance(k, str):
            raise TypeError(f"schema.json の series.{series_name}.keys に文字列以外が含まれています")
        cols.append(k)
    return cols


__all__ = ["extract_result_files", "series_columns"]



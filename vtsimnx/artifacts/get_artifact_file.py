from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import requests

from vtsimnx.run_calc._http import _api_key_headers
from ._schema import extract_result_files, series_columns


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key is not None:
        key = api_key.strip()
        return key or None
    env_key = os.getenv("VTSIMNX_API_KEY", "").strip()
    return env_key or None


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    return _api_key_headers(_resolve_api_key(api_key))


def _get_manifest(
    base_url: str,
    artifact_dir: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/manifest"
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    obj = resp.json()
    if not isinstance(obj, dict):
        raise TypeError(f"manifest response must be dict, got {type(obj).__name__}")
    return obj


def _filename_to_key_map(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    manifest から「ファイル名 -> download key」の対応を作る。
    """
    mapping: Dict[str, str] = {"manifest.json": "manifest"}
    out = manifest.get("output") if isinstance(manifest.get("output"), dict) else {}

    log_file = out.get("log_file") if isinstance(out, dict) else None
    if isinstance(log_file, str) and log_file:
        mapping[log_file] = "log"
    else:
        mapping["solver.log"] = "log"

    builder_log = out.get("builder_log_file") if isinstance(out, dict) else None
    if isinstance(builder_log, str) and builder_log:
        mapping[builder_log] = "builder_log"
    else:
        mapping.setdefault("builder.log", "builder_log")

    result_files: Dict[str, str] = {}
    try:
        result_files = extract_result_files(manifest)
    except ValueError:
        result_files = {}
    for key, filename in result_files.items():
        if isinstance(filename, str) and filename:
            mapping[filename] = key

    files = manifest.get("files")
    if isinstance(files, dict):
        for key, filename in files.items():
            if isinstance(key, str) and isinstance(filename, str) and filename:
                mapping[filename] = key

    return mapping


def _resolve_download_key(manifest: Dict[str, Any], name_or_filename: str) -> str:
    raw = name_or_filename.strip().lstrip("/")
    basename = raw.split("/")[-1]
    mapping = _filename_to_key_map(manifest)
    known_keys = set(mapping.values())

    if basename in mapping:
        return mapping[basename]
    if raw in known_keys:
        return raw
    if basename in known_keys:
        return basename
    raise KeyError(
        f"artifact key/filename not found in manifest whitelist: {name_or_filename!r}"
    )


def _download_by_key(
    base_url: str,
    artifact_dir: str,
    key: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> bytes:
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/download/{key}"
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _load_json_bytes(raw: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid json bytes: {e}") from e
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object, got {type(obj).__name__}")
    return obj


def _infer_index_spec_from_manifest(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    manifest（または manifest["output"]）に index があれば返す。
    """
    output = manifest.get("output") if isinstance(manifest.get("output"), dict) else manifest
    if isinstance(output, dict) and isinstance(output.get("index"), dict):
        return output.get("index")  # type: ignore[return-value]
    return None


def _apply_time_index_inplace(df: "pd.DataFrame", index_spec: Dict[str, Any], *, expected_length: int) -> None:
    """
    index_spec（start/timestep/length）に基づいて df.index を time軸にする。
    失敗しても例外は上位で握りつぶす前提。
    """
    start = index_spec.get("start")
    timestep = index_spec.get("timestep")
    length = index_spec.get("length")
    if not (isinstance(start, str) and isinstance(timestep, int) and isinstance(length, int)):
        return
    if length != expected_length:
        return

    start_ts = pd.to_datetime(start)
    if timestep == 0:
        df.index = pd.DatetimeIndex([start_ts] * expected_length)
    else:
        df.index = pd.date_range(start=start_ts, periods=expected_length, freq=pd.to_timedelta(timestep, unit="s"))
    df.index.name = "time"


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
    """
    if manifest is None:
        manifest = _get_manifest(base_url, artifact_dir, timeout=timeout, api_key=api_key)
    key = _resolve_download_key(manifest, filename)
    data = _download_by_key(base_url, artifact_dir, key, timeout=timeout, api_key=api_key)
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)
    return data


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
    - それ以外は取得したバイト列を返す
    """
    manifest = _get_manifest(base_url, artifact_dir, timeout=timeout, api_key=api_key)
    key = _resolve_download_key(manifest, filename)
    data = _download_by_key(base_url, artifact_dir, key, timeout=timeout, api_key=api_key)

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    result_files = extract_result_files(manifest)
    bin_basename = result_files.get(key) if key in result_files else filename.split("/")[-1]
    if not isinstance(bin_basename, str):
        bin_basename = filename.split("/")[-1]

    if not bin_basename.endswith(".f32.bin"):
        return data

    schema_raw = _download_by_key(base_url, artifact_dir, "schema", timeout=timeout, api_key=api_key)
    schema = _load_json_bytes(schema_raw)

    dtype = schema.get("dtype")
    layout = schema.get("layout")
    if dtype != "f32le":
        raise ValueError(f"schema.json dtype が想定外です: {dtype!r} (想定: 'f32le')")
    if layout != "timestep-major":
        raise ValueError(f"schema.json layout が想定外です: {layout!r} (想定: 'timestep-major')")

    T = schema.get("length")
    if not isinstance(T, int) or T < 0:
        raise ValueError(f"schema.json length が不正です: {T!r}")

    series_name: Optional[str] = None
    for k, v in result_files.items():
        if v == bin_basename:
            series_name = k
            break
    if series_name is None and key in result_files:
        series_name = key
    if series_name is None:
        raise KeyError(f"manifest.json から {bin_basename} に対応する series 名が見つかりませんでした")

    cols = series_columns(schema, series_name)
    N = len(cols)

    arr = np.frombuffer(data, dtype=np.dtype("<f4"))
    expected = T * N
    if arr.size != expected:
        raise ValueError(f"{bin_basename}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})")
    arr = arr.reshape((T, N))

    df = pd.DataFrame(arr, columns=cols)
    # index_spec が明示されていればそれを優先。無ければ manifest/output の index を使う。
    if index_spec is None:
        index_spec = _infer_index_spec_from_manifest(manifest)

    if isinstance(index_spec, dict):
        try:
            _apply_time_index_inplace(df, index_spec, expected_length=T)
        except (TypeError, ValueError):
            # index付与に失敗してもDataFrame自体は返す
            pass
    return df

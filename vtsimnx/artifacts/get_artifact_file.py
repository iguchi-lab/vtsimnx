from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

from ._schema import extract_result_files, series_columns


def _get_bytes(base_url: str, artifact_dir: str, relpath: str, timeout: float) -> bytes:
    url = base_url.rstrip("/") + f"/work/{artifact_dir}/{relpath.lstrip('/')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _get_bytes_fallback(base_url: str, artifact_dir: str, relpaths: List[str], timeout: float) -> bytes:
    last_exc: Optional[Exception] = None
    for p in relpaths:
        try:
            return _get_bytes(base_url, artifact_dir, p, timeout=timeout)
        except Exception as e:
            last_exc = e
    raise last_exc  # type: ignore[misc]


def _get_artifact_bytes_with_used_path(
    base_url: str,
    artifact_dir: str,
    filename: str,
    *,
    timeout: float,
) -> tuple[bytes, str]:
    # まずは指定されたパスで取得（ダメなら artifacts/ 配下も試す）
    data: Optional[bytes] = None
    tried: List[str] = []
    last_exc: Optional[Exception] = None
    for rel in [filename, f"artifacts/{filename}" if not filename.startswith("artifacts/") else filename]:
        if rel in tried:
            continue
        tried.append(rel)
        try:
            data = _get_bytes(base_url, artifact_dir, rel, timeout=timeout)
            return data, rel
        except Exception as e:
            last_exc = e
            data = None
    raise last_exc  # type: ignore[misc]


def get_artifact_bytes(
    base_url: str,
    artifact_dir: str,
    filename: str,
    *,
    output_path: Optional[str] = None,
    timeout: float = 60.0,
) -> bytes:
    """
    成果物ディレクトリからファイルを1つ取得して bytes を返す（復元はしない）。

    想定API:
      GET {base_url}/work/{artifact_dir}/{filename}

    - filename が見つからない場合は artifacts/filename もフォールバックで試す
    - output_path を指定すると保存も行う
    """
    data, _used = _get_artifact_bytes_with_used_path(
        base_url, artifact_dir, filename, timeout=timeout
    )
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)
    return data


def get_artifact_file(
    base_url: str,
    artifact_dir: str,
    filename: str,
    output_path: Optional[str] = None,
    timeout: float = 60.0,
) -> Union[bytes, "pd.DataFrame"]:
    """
    成果物ディレクトリからファイルを1つ取得する。

    想定API:
      GET {base_url}/work/{artifact_dir}/{filename}

    - output_path を指定すると、取得した内容をそのパスに保存する（Noneなら保存しない）
    - `.f32.bin` の場合は `schema.json` と `manifest.json` を参照して DataFrame に復元して返す
      - dtype: schema.json の "f32le" -> np.dtype("<f4")
      - layout: schema.json の "timestep-major" -> shape=(T, N)
    - それ以外は取得したバイト列を返す
    """
    def _load_json_bytes(raw: bytes) -> Dict[str, Any]:
        return json.loads(raw.decode("utf-8"))

    data, used_relpath = _get_artifact_bytes_with_used_path(
        base_url, artifact_dir, filename, timeout=timeout
    )
    filename = used_relpath

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    # 自動復元: *.f32.bin は DataFrame を返す
    if not filename.endswith(".f32.bin"):
        return data

    # schema/manifest は配置ゆれがあるので両方試す
    schema = _load_json_bytes(_get_bytes_fallback(base_url, artifact_dir, ["schema.json", "artifacts/schema.json"], timeout=timeout))
    manifest = _load_json_bytes(_get_bytes_fallback(base_url, artifact_dir, ["manifest.json", "artifacts/manifest.json"], timeout=timeout))

    dtype = schema.get("dtype")
    layout = schema.get("layout")
    if dtype != "f32le":
        raise ValueError(f"schema.json dtype が想定外です: {dtype!r} (想定: 'f32le')")
    if layout != "timestep-major":
        raise ValueError(f"schema.json layout が想定外です: {layout!r} (想定: 'timestep-major')")

    T = schema.get("length")
    if not isinstance(T, int) or T < 0:
        raise ValueError(f"schema.json length が不正です: {T!r}")

    result_files = extract_result_files(manifest)
    bin_basename = filename.split("/")[-1]

    # manifest: series_name -> bin_filename なので逆引き
    series_name: Optional[str] = None
    for k, v in result_files.items():
        if v == bin_basename:
            series_name = k
            break
    if series_name is None:
        raise KeyError(f"manifest.json から {bin_basename} に対応する series 名が見つかりませんでした")

    cols = series_columns(schema, series_name)
    N = len(cols)

    arr = np.frombuffer(data, dtype=np.dtype("<f4"))
    expected = T * N
    if arr.size != expected:
        raise ValueError(f"{bin_basename}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})")
    arr = arr.reshape((T, N))

    return pd.DataFrame(arr, columns=cols)



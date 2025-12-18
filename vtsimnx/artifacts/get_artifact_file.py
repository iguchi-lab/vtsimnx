from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests


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
    def _get_bytes(relpath: str) -> bytes:
        url = base_url.rstrip("/") + f"/work/{artifact_dir}/{relpath.lstrip('/')}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content

    def _get_bytes_fallback(relpaths: List[str]) -> bytes:
        last_exc: Optional[Exception] = None
        for p in relpaths:
            try:
                return _get_bytes(p)
            except Exception as e:
                last_exc = e
        raise last_exc  # type: ignore[misc]

    def _load_json_bytes(raw: bytes) -> Dict[str, Any]:
        return json.loads(raw.decode("utf-8"))

    def _extract_result_files(manifest: Dict[str, Any]) -> Dict[str, str]:
        # 想定される形（いずれか）:
        #   - {"result": {"result_files": {...}}}
        #   - {"result_files": {...}}
        #   - {"files": {...}}
        if isinstance(manifest.get("result"), dict) and isinstance(manifest["result"].get("result_files"), dict):
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

    def _series_columns(schema: Dict[str, Any], series_name: str) -> List[str]:
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

    # まずは指定されたパスで取得（ダメなら artifacts/ 配下も試す）
    data: Optional[bytes] = None
    tried: List[str] = []
    last_exc: Optional[Exception] = None
    for rel in [filename, f"artifacts/{filename}" if not filename.startswith("artifacts/") else filename]:
        if rel in tried:
            continue
        tried.append(rel)
        try:
            data = _get_bytes(rel)
            filename = rel  # 実際に取得できた相対パスに寄せる
            break
        except Exception as e:
            last_exc = e
            data = None

    if data is None:
        # 最後の例外をそのまま出す（requests の raise_for_status など）
        raise last_exc  # type: ignore[misc]

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    # 自動復元: *.f32.bin は DataFrame を返す
    if not filename.endswith(".f32.bin"):
        return data

    # schema/manifest は配置ゆれがあるので両方試す
    schema = _load_json_bytes(_get_bytes_fallback(["schema.json", "artifacts/schema.json"]))
    manifest = _load_json_bytes(_get_bytes_fallback(["manifest.json", "artifacts/manifest.json"]))

    dtype = schema.get("dtype")
    layout = schema.get("layout")
    if dtype != "f32le":
        raise ValueError(f"schema.json dtype が想定外です: {dtype!r} (想定: 'f32le')")
    if layout != "timestep-major":
        raise ValueError(f"schema.json layout が想定外です: {layout!r} (想定: 'timestep-major')")

    T = schema.get("length")
    if not isinstance(T, int) or T < 0:
        raise ValueError(f"schema.json length が不正です: {T!r}")

    result_files = _extract_result_files(manifest)
    bin_basename = filename.split("/")[-1]

    # manifest: series_name -> bin_filename なので逆引き
    series_name: Optional[str] = None
    for k, v in result_files.items():
        if v == bin_basename:
            series_name = k
            break
    if series_name is None:
        raise KeyError(f"manifest.json から {bin_basename} に対応する series 名が見つかりませんでした")

    cols = _series_columns(schema, series_name)
    N = len(cols)

    arr = np.frombuffer(data, dtype=np.dtype("<f4"))
    expected = T * N
    if arr.size != expected:
        raise ValueError(f"{bin_basename}: 要素数が不一致です (actual={arr.size}, expected={expected}, T={T}, N={N})")
    arr = arr.reshape((T, N))

    return pd.DataFrame(arr, columns=cols)



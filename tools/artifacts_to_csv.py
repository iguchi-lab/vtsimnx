from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_result_files(manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    manifest.json から「系列名 -> ファイル名」の辞書を取り出す。

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

    # keys が [] の場合はスカラー
    if len(keys) == 0:
        return [series_name]

    cols: List[str] = []
    for k in keys:
        if not isinstance(k, str):
            raise TypeError(f"schema.json の series.{series_name}.keys に文字列以外が含まれています")
        cols.append(k)
    return cols


def _read_f32le_timestep_major(bin_path: Path, T: int, N: int) -> np.ndarray:
    data = np.fromfile(bin_path, dtype=np.dtype("<f4"))
    expected = T * N
    if data.size != expected:
        raise ValueError(f"{bin_path.name}: 要素数が不一致です (actual={data.size}, expected={expected}, T={T}, N={N})")
    return data.reshape((T, N))


def _iter_f32_bins(result_files: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    manifestの result_files から、*.f32.bin のみを（series_name, filename）で列挙する。
    """
    out: List[Tuple[str, str]] = []
    for series_name, fname in result_files.items():
        if not fname.endswith(".f32.bin"):
            continue
        out.append((series_name, fname))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="work/output.artifacts.XXXX の *.f32.bin を schema.json に基づきCSVへ変換します")
    parser.add_argument("--artifact-dir", required=True, help="例: work/output.artifacts.XXXX")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifacts_dir = artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"artifacts ディレクトリが見つかりません: {artifacts_dir}")

    # schema/manifest は配置ゆれがあるため artifacts/直下 or artifact_dir直下を許容
    manifest_candidates = [artifacts_dir / "manifest.json", artifact_dir / "manifest.json"]
    schema_candidates = [artifacts_dir / "schema.json", artifact_dir / "schema.json"]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    schema_path = next((p for p in schema_candidates if p.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"manifest.json が見つかりません: {', '.join(str(p) for p in manifest_candidates)}")
    if schema_path is None:
        raise FileNotFoundError(f"schema.json が見つかりません: {', '.join(str(p) for p in schema_candidates)}")

    manifest = _load_json(manifest_path)
    schema = _load_json(schema_path)

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
    pairs = _iter_f32_bins(result_files)
    if len(pairs) == 0:
        print("変換対象の *.f32.bin が見つかりませんでした（manifest.json を確認してください）")
        return 0

    for series_name, bin_name in pairs:
        # バイナリも artifacts/直下 or artifact_dir直下を許容
        bin_candidates = [artifacts_dir / bin_name, artifact_dir / bin_name]
        bin_path = next((p for p in bin_candidates if p.exists()), None)
        if bin_path is None:
            raise FileNotFoundError(f"バイナリが見つかりません: {', '.join(str(p) for p in bin_candidates)}")

        cols = _series_columns(schema, series_name)
        N = len(cols)

        arr = _read_f32le_timestep_major(bin_path, T=T, N=N)
        df = pd.DataFrame(arr, columns=cols)

        csv_name = bin_name[: -len(".f32.bin")] + ".csv"
        out_path = artifacts_dir / csv_name
        df.to_csv(out_path, index=False)
        print(f"OK: {bin_name} -> {out_path.name} (shape={arr.shape})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from vtsimnx.artifacts._schema import extract_result_files, series_columns


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONの解析に失敗しました: {path}") from e
    except OSError as e:
        raise OSError(f"ファイルの読み込みに失敗しました: {path}") from e


def _find_first_existing(candidates: List[Path], *, kind: str) -> Path:
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"{kind} が見つかりません: {', '.join(str(p) for p in candidates)}")
    return found


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
    manifest_path = _find_first_existing(manifest_candidates, kind="manifest.json")
    schema_path = _find_first_existing(schema_candidates, kind="schema.json")

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

    result_files = extract_result_files(manifest)
    pairs = _iter_f32_bins(result_files)
    if len(pairs) == 0:
        print("変換対象の *.f32.bin が見つかりませんでした（manifest.json を確認してください）")
        return 0

    for series_name, bin_name in pairs:
        # バイナリも artifacts/直下 or artifact_dir直下を許容
        bin_candidates = [artifacts_dir / bin_name, artifact_dir / bin_name]
        bin_path = _find_first_existing(bin_candidates, kind="バイナリ")

        cols = series_columns(schema, series_name)
        N = len(cols)

        arr = _read_f32le_timestep_major(bin_path, T=T, N=N)
        df = pd.DataFrame(arr, columns=cols)

        csv_name = bin_name[: -len(".f32.bin")] + ".csv"
        out_path = artifacts_dir / csv_name
        try:
            df.to_csv(out_path, index=False)
        except OSError as e:
            raise OSError(f"CSVの書き込みに失敗しました: {out_path}") from e
        print(f"OK: {bin_name} -> {out_path.name} (shape={arr.shape})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



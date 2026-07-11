from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from vtsimnx.artifacts import decode_f32_series, extract_result_files
from vtsimnx.artifacts._decode import load_json_bytes


def _load_json(path: Path) -> dict:
    try:
        return load_json_bytes(path.read_bytes())
    except OSError as e:
        raise OSError(f"ファイルの読み込みに失敗しました: {path}") from e


def _find_first_existing(candidates: List[Path], *, kind: str) -> Path:
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"{kind} が見つかりません: {', '.join(str(p) for p in candidates)}")
    return found


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
    parser = argparse.ArgumentParser(
        description="work/output.artifacts.XXXX の *.f32.bin を schema.json に基づきCSVへ変換します"
    )
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

    result_files = extract_result_files(manifest)
    pairs = _iter_f32_bins(result_files)
    if len(pairs) == 0:
        print("変換対象の *.f32.bin が見つかりませんでした（manifest.json を確認してください）")
        return 0

    for series_name, bin_name in pairs:
        # バイナリも artifacts/直下 or artifact_dir直下を許容
        bin_candidates = [artifacts_dir / bin_name, artifact_dir / bin_name]
        bin_path = _find_first_existing(bin_candidates, kind="バイナリ")
        raw = bin_path.read_bytes()
        df = decode_f32_series(raw, schema, series_name, source_name=bin_name)

        csv_name = bin_name[: -len(".f32.bin")] + ".csv"
        out_path = artifacts_dir / csv_name
        try:
            df.to_csv(out_path, index=False)
        except OSError as e:
            raise OSError(f"CSVの書き込みに失敗しました: {out_path}") from e
        print(f"OK: {bin_name} -> {out_path.name} (shape={df.shape})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

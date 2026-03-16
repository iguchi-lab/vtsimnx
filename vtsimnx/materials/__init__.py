"""
materials パッケージ（窓口）。

`vtsimnx.materials` は「材料物性テーブル（dict）」として公開する。
既存の table 定義をベースに、materials 配下の CSV（旧/新フォーマット）を
読める場合は追加マージする。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .table import materials as _table_materials


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """
    UTF-8(BOM含む) を優先し、失敗時は cp932 で再読込する。
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932")


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    name = str(value).strip()
    # 旧CSVに混入しやすい先頭引用符を除去
    return name.lstrip("'").strip()


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_materials_from_csv() -> Dict[str, Dict[str, float]]:
    """
    materials フォルダ直下の CSV を走査し、lambda/v_capa を持つ行のみ取り込む。
    - 旧形式: 熱伝導率 + 容積比熱[kJ/m3K]
    - 新形式: 熱伝導率 + 比熱[kJ/kgK] + 密度[kg/m3]
    """
    csv_materials: Dict[str, Dict[str, float]] = {}
    base_dir = Path(__file__).resolve().parent

    for path in sorted(base_dir.glob("*.csv")):
        try:
            df = _read_csv_with_fallback(path)
        except Exception:
            # CSVが壊れていても既存 table のみで動作を継続する
            continue

        if "材料名" not in df.columns:
            continue

        # 旧フォーマット: 容積比熱が直接入っている
        old_cols = {"熱伝導率 [W/mK]", "容積比熱 [KJ/m3･K]"}
        if old_cols.issubset(df.columns):
            for _, row in df.iterrows():
                name = _normalize_name(row.get("材料名"))
                lam = _to_float(row.get("熱伝導率 [W/mK]"))
                v_capa_kj = _to_float(row.get("容積比熱 [KJ/m3･K]"))
                if not name or lam is None or v_capa_kj is None:
                    continue
                if lam <= 0.0 or v_capa_kj <= 0.0:
                    continue
                csv_materials[name] = {"lambda": lam, "v_capa": v_capa_kj * 1000.0}
            continue

        # 新フォーマット: 比熱[kJ/kgK] と密度[kg/m3] から容積熱容量を算出
        new_cols = {"熱伝導率 [W/mK]", "比熱 [KJ/kg･K]", "密度 [kg/m3]"}
        if new_cols.issubset(df.columns):
            for _, row in df.iterrows():
                name = _normalize_name(row.get("材料名"))
                lam = _to_float(row.get("熱伝導率 [W/mK]"))
                cp_kj = _to_float(row.get("比熱 [KJ/kg･K]"))
                rho = _to_float(row.get("密度 [kg/m3]"))
                if not name or lam is None or cp_kj is None or rho is None:
                    continue
                if lam <= 0.0 or cp_kj <= 0.0 or rho <= 0.0:
                    continue
                csv_materials[name] = {"lambda": lam, "v_capa": cp_kj * rho * 1000.0}

    return csv_materials


# CSV を取り込んだ上で table を後勝ちにし、table.py を優先
materials: Dict[str, Dict[str, float]] = {
    **_load_materials_from_csv(),
    **_table_materials,
}

__all__ = ["materials"]



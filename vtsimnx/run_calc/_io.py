from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Union


def _write_json(path: Union[str, Path], obj: Any) -> None:
    """
    obj を JSONとして保存する（.gz の場合は gzip 圧縮）。
    """
    p = Path(path)
    if p.suffix.lower() == ".gz":
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


__all__ = ["_write_json"]



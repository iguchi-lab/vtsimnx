from __future__ import annotations

from typing import Any, Dict, Optional
import json

import requests


def list_artifact_files(
    base_url: str,
    artifact_dir: str,
    output_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    成果物ディレクトリのファイル一覧を取得する。

    想定API:
      GET {base_url}/artifacts/{artifact_dir}/files

    output_path を指定すると、取得したJSONを保存する（Noneなら保存しない）。
    """
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/files"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return data  # type: ignore[return-value]



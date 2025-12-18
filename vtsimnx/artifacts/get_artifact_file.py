from __future__ import annotations

from typing import Optional

import requests


def get_artifact_file(
    base_url: str,
    artifact_dir: str,
    filename: str,
    output_path: Optional[str] = None,
    timeout: float = 60.0,
) -> bytes:
    """
    成果物ディレクトリからファイルを1つ取得する。

    想定API:
      GET {base_url}/artifacts/{artifact_dir}/files/{filename}

    - output_path を指定すると、取得した内容をそのパスに保存する（Noneなら保存しない）
    - 返り値は取得したバイト列
    """
    url = base_url.rstrip("/") + f"/artifacts/{artifact_dir}/files/{filename}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    return data



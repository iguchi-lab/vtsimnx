from __future__ import annotations

from typing import Any, Dict


def _output_block(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    APIレスポンスの形の揺れを吸収: {"output": {...}} / {"result": {...}} / 直下
    """
    if isinstance(resp_json.get("output"), dict):
        return resp_json["output"]  # type: ignore[return-value]
    if isinstance(resp_json.get("result"), dict):
        return resp_json["result"]  # type: ignore[return-value]
    return resp_json


__all__ = ["_output_block"]



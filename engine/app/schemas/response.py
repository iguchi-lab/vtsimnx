"""API レスポンスモデル。"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SimulationResponse(BaseModel):
    """
    ソルバの計算結果を表すデータモデル。

    - result: ソルバから返却される任意の JSON 互換オブジェクト
    """

    result: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    warning_details: List[Dict[str, Any]] = Field(default_factory=list)

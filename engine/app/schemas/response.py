"""API レスポンスモデル。"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SimulationResponse(BaseModel):
    """
    ソルバの計算結果を表すデータモデル。

    - result: ソルバから返却される任意の JSON 互換オブジェクト
    """

    result: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    warning_details: List[Dict[str, Any]] = Field(default_factory=list)


class Issue(BaseModel):
    """検査・展開 API の統一指摘。path は JSON ポインタに近い配列。"""

    code: str
    message: str
    path: List[Any] = Field(default_factory=list)
    severity: Literal["error", "warning"] = "warning"
    hint: Optional[str] = None


class ValidateResponse(BaseModel):
    ok: bool
    issues: List[Issue] = Field(default_factory=list)


class BuildStats(BaseModel):
    nodes: int
    thermal_branches: int
    ventilation_branches: int


class BuildResponse(BaseModel):
    ok: bool
    issues: List[Issue] = Field(default_factory=list)
    stats: Optional[BuildStats] = None
    config: Optional[Dict[str, Any]] = None

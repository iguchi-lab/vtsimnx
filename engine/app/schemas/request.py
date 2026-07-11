"""API リクエストモデル。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.schemas.config import RawSimConfig, UnknownKeysMode


class SimulationRequest(BaseModel):
    """
    ソルバに渡す入力設定を表すデータモデル。

    - config: ユーザー入力JSON（raw）。API側で builder により正規化/展開してから C++ ソルバに渡す。
    - unknown_keys: 未知フィールドの扱い（strip=警告して削除 / error=422）
    """

    config: RawSimConfig
    unknown_keys: UnknownKeysMode = "strip"
    debug: bool = False
    debug_verbosity: int = 2
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None

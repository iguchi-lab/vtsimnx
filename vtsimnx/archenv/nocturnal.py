from __future__ import annotations

import pandas as pd
import numpy as np

from .archenv import e, T, MJ_to_Wh, Sigma


# 夜間放射 MJ/m2
rn = lambda t, h: (94.21 + 39.06 * np.sqrt(e(t, h) / 100) \
                   - 0.85 * Sigma * np.power(T(t), 4)) * 4.187 / 1000


def nocturnal_gain_by_angles(
    *,
    傾斜角: float,
    外気温: pd.Series | None = None,
    外気相対湿度: pd.Series | None = None,
    夜間放射量_水平: pd.Series | None = None,
    名前: str = "任意面",
) -> pd.DataFrame:
    """
    傾斜角だけ指定して、その面の夜間放射量（長波放射）を返す。

    入力（どちらか）:
      - 外気温 + 外気相対湿度: rn(t,h) から水平面の夜間放射量を推算
      - 夜間放射量_水平: 水平面の夜間放射量 [Wh/m2] を直接与える

    傾斜角:
      0=水平上向き, 90=鉛直

    モデル:
      旧 make_nocturnal の vertical_factor=0.5 を一般化し、
      等方天空の view factor を用いて
        (F_sky = (1+cosβ)/2)
      で水平面の夜間放射量をスケールする。
      （β=0 → 1.0, β=90 → 0.5）
    """
    if 夜間放射量_水平 is None:
        if 外気温 is None or 外気相対湿度 is None:
            raise TypeError("nocturnal_gain_by_angles: 外気温/外気相対湿度 か 夜間放射量_水平 を指定してください。")
        夜間放射量_水平 = MJ_to_Wh(rn(外気温, 外気相対湿度))

    beta = np.radians(float(傾斜角))
    f_sky = (1.0 + float(np.cos(beta))) / 2.0

    out = pd.DataFrame(index=夜間放射量_水平.index)
    out["夜間放射量_水平"] = 夜間放射量_水平
    out[f"夜間放射量（{名前}）"] = 夜間放射量_水平 * f_sky
    return out

__all__ = ["rn", "nocturnal_gain_by_angles"]



from __future__ import annotations

import pandas as pd
import numpy as np

from .archenv import e, T, MJ_to_Wh, Sigma


# 夜間放射 MJ/m2
rn = lambda t, h: (94.21 + 39.06 * np.sqrt(e(t, h) / 100) \
                   - 0.85 * Sigma * np.power(T(t), 4)) * 4.187 / 1000


def make_nocturnal(*args, vertical_factor: float = 0.5, **kwargs):
    """夜間放射量を算出する

    指定方法:
      - (DataFrame) を位置引数で与える: 列名から自動マッピング
          - 外気温: '外気温'
          - 外気相対湿度: '外気相対湿度'
          - 夜間放射量: '夜間放射量' / 'n_r'
      - 外気温/外気相対湿度 を与える: 気温・相対湿度から推算
      - 夜間放射量（'夜間放射量' or 'n_r'）を与える: その値を使用

    戻り値:
      (DataFrame) 夜間放射量（列: '夜間放射量', '夜間_水平', '夜間_垂直'）
    """
    # 互換: 先頭位置引数で DataFrame/Series を受けた場合に自動でキーへマッピングする
    if args:
        first = args[0]
        if isinstance(first, pd.DataFrame):
            df0 = first
            if "夜間放射量" in df0.columns:
                kwargs.setdefault("夜間放射量", df0["夜間放射量"])
            elif "n_r" in df0.columns:
                kwargs.setdefault("n_r", df0["n_r"])
            else:
                # 推算用
                if "外気温" in df0.columns:
                    kwargs.setdefault("外気温", df0["外気温"])
                if "外気相対湿度" in df0.columns:
                    kwargs.setdefault("外気相対湿度", df0["外気相対湿度"])
        elif isinstance(first, pd.Series):
            # Series だけ渡された場合は夜間放射量として扱う
            kwargs.setdefault("夜間放射量", first)

    def _build_df(n_r: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=n_r.index)
        # 後方互換
        df["夜間放射量"] = n_r
        # 日射と同様に「水平/垂直」を明示した列も返す
        df["夜間放射量_水平"] = n_r
        df["夜間放射量_垂直"] = n_r * float(vertical_factor)
        return df

    if "外気温" in kwargs:
        t = kwargs["外気温"]
        h = kwargs["外気相対湿度"]
        n_r = MJ_to_Wh(rn(t, h))
        return _build_df(n_r)

    if "夜間放射量" in kwargs:
        n_r = kwargs["夜間放射量"]
        return _build_df(n_r)

    if "n_r" in kwargs:
        n_r = kwargs["n_r"]
        return _build_df(n_r)

    raise TypeError("make_nocturnal: 外気温/外気相対湿度 か、夜間放射量（夜間放射量 or n_r）を指定してください。")

__all__ = ["rn", "make_nocturnal"]



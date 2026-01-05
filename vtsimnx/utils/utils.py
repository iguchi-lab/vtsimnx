import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

###############################################################################
# public functions (backward compatible names)
###############################################################################
def read_csv(fn: Union[str, Path]) -> pd.DataFrame:
    """
    CSV を読み込み、日時列をパースしつつ、欠損を前方/後方補間で補完した DataFrame を返す。
    先頭列を index として扱う。
    """
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    # 先に後方補完 → 前方補完の順でギャップを埋める
    return df.fillna(method="bfill").fillna(method="ffill")


def index(freq: str, length: int) -> pd.DatetimeIndex:
    """
    開始時刻 2026-01-01 01:00:00 から、秒数 length の範囲を freq 間隔で生成した DatetimeIndex を返す。
    """
    start = datetime(2026, 1, 1, 1, 0, 0)
    end = start + timedelta(seconds=length - 1)
    return pd.date_range(start=start, end=end, freq=freq)


###############################################################################
# helpers
###############################################################################
def encode(obj: Any):
    """
    JSON 互換へエンコードするための簡易ヘルパー。
    pandas の DatetimeIndex / Series と numpy 配列・スカラに対応。
    """
    if isinstance(obj, pd.DatetimeIndex):
        return obj.strftime("%Y/%m/%d %H:%M:%S").to_list()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def read_json(fn: Union[str, Path]):
    """
    JSON ファイルを読み込み、Python オブジェクト（dict など）を返す。
    """
    p = Path(fn)
    if p.suffix.lower() == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_hasp(fn: Union[str, Path]) -> pd.DataFrame:
    """
    HASP 形式（1 日あたり 7 行、各行 72 文字×24 時間の 3 桁数値）の気象データを読み込み、
    列 ['t_ex','h_ex','i_b','i_d','n_r','w_d','w_s'] を持つ DataFrame を返す。

    単位変換:
      - 外気温: (value - 500) / 10  [℃]
      - 外気絶対湿度: value / 10          [g/kg']
      - 直達日射量, 水平面拡散日射量, 夜間放射量: ×1.16222   [kcal/(m2・h) → W/m2]
      - 風向: 風向（整数カテゴリ）
      - 風速: value / 10           [m/s]
    """
    # HASP の i_b / i_d は「直達（法線面相当） / 水平面拡散」を想定
    clm = ["外気温", "外気絶対湿度", "直達日射量", "水平面拡散日射量", "夜間放射量", "風向", "風速"]
    str_dat = [""] * 7

    with open(fn, "rb") as f:
        # 行単位で読み込み（ASCII 数字前提）
        dat = [line.decode() for line in f.readlines()]

    # 365 日分 × 7 行を 24 時間×3 桁で連結
    for day in range(365):
        base = day * 7
        for i in range(7):
            str_dat[i] += dat[base + i][:72]

    df = pd.DataFrame()
    for i, name in enumerate(clm):
        # 3 桁ずつ整数化して 24*365 要素へ
        df[name] = [int(str_dat[i][j * 3 : j * 3 + 3]) for j in range(24 * 365)]

    # 単位変換
    df["外気温"] = (df["外気温"] - 500) / 10
    df["外気絶対湿度"] = df["外気絶対湿度"] / 10
    df["直達日射量"] = df["直達日射量"] * 1.16222
    df["水平面拡散日射量"] = df["水平面拡散日射量"] * 1.16222
    df["夜間放射量"] = df["夜間放射量"] * 1.16222
    df["風向"] = df["風向"].astype(int)
    # 0:無風, 1:NNE, 2:NE, 3:ENE, 4:E, 5:ESE, 6:SE, 7:SSE, 8:S,
    # 9:SSW, 10:SW, 11:WSW, 12:W, 13:WNW, 14:NW, 15:NNW, 16:N
    df["風速"] = df["風速"] / 10
    df.index = index('1h', 365 * 24 * 3600)
    return df
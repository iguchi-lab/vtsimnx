import numpy as np
import pandas as pd


def make_wind(d, s, c_in: float = 0.7, c_out: float = -0.55, c_horizontal: float = -0.90):
    """風向・風速から各方位の風圧を算出する

    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数

    戻り値:
      (DataFrame, dict[str, Series]) 風向・風速などの中間列と、方位別風圧（E/S/W/N/H）
    """
    if not isinstance(d, pd.Series) or not isinstance(s, pd.Series):
        raise TypeError("d と s は pandas.Series で指定してください。")
    if not d.index.equals(s.index):
        raise ValueError("d と s の index は一致している必要があります。")
    if d.isna().any() or s.isna().any():
        raise ValueError("d と s に NaN は指定できません。")
    if (s < 0).any():
        raise ValueError("風速 s は 0 以上である必要があります。")

    # 風向カテゴリは 0..16（0:無風, 1..16:方位）
    d_num = pd.to_numeric(d, errors="coerce")
    if d_num.isna().any():
        raise TypeError("風向カテゴリ d は数値（0..16）で指定してください。")
    if ((d_num < 0) | (d_num > 16)).any():
        raise ValueError("風向カテゴリ d は 0..16 の範囲で指定してください。")

    d = d_num.astype("float64")
    s = s.astype("float64")
    df = pd.DataFrame(index=d.index)

    df["風速_E"] = np.sin(np.radians(d * 22.5)) * s
    df["風速_S"] = -np.cos(np.radians(d * 22.5)) * s
    df["風速_W"] = -np.sin(np.radians(d * 22.5)) * s
    df["風速_N"] = np.cos(np.radians(d * 22.5)) * s

    df.loc[df["風速_E"] >= 0, "風圧_E"] = 1.2 / 2 * c_in * df["風速_E"] ** 2
    df.loc[df["風速_E"] < 0, "風圧_E"] = -1.2 / 2 * c_out * df["風速_E"] ** 2
    df.loc[df["風速_S"] >= 0, "風圧_S"] = 1.2 / 2 * c_in * df["風速_S"] ** 2
    df.loc[df["風速_S"] < 0, "風圧_S"] = -1.2 / 2 * c_out * df["風速_S"] ** 2
    df.loc[df["風速_W"] >= 0, "風圧_W"] = 1.2 / 2 * c_in * df["風速_W"] ** 2
    df.loc[df["風速_W"] < 0, "風圧_W"] = -1.2 / 2 * c_out * df["風速_W"] ** 2
    df.loc[df["風速_N"] >= 0, "風圧_N"] = 1.2 / 2 * c_in * df["風速_N"] ** 2
    df.loc[df["風速_N"] < 0, "風圧_N"] = -1.2 / 2 * c_out * df["風速_N"] ** 2
    df["風圧_H"] = 1.2 / 2 * c_horizontal * (s**2)

    wind_pressure = {
        "E": df["風圧_E"],
        "S": df["風圧_S"],
        "W": df["風圧_W"],
        "N": df["風圧_N"],
        "H": df["風圧_H"],
    }

    return df, wind_pressure


__all__ = ["make_wind"]



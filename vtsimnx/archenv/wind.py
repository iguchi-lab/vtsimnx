import pandas as pd
import numpy as np


def make_wind(d, s, c_in = 0.7, c_out = -0.55, c_horizontal = -0.90):
    """風向・風速から各方位の風圧を算出する
    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数
    戻り値:
      (DataFrame, dict[str, Series]) 風向・風速などの中間列と、方位別風圧（E/S/W/N/H）
    """
    df = pd.DataFrame(index = d.index)

    df['風速_E'] =   np.sin(np.radians(d * 22.5)) * s
    df['風速_S'] = - np.cos(np.radians(d * 22.5)) * s
    df['風速_W'] = - np.sin(np.radians(d * 22.5)) * s
    df['風速_N'] =   np.cos(np.radians(d * 22.5)) * s

    df.loc[df['風速_E'] >= 0, '風圧_E'] =  1.2 / 2 * c_in  * df['風速_E'] ** 2
    df.loc[df['風速_E'] < 0,  '風圧_E'] = -1.2 / 2 * c_out * df['風速_E'] ** 2
    df.loc[df['風速_S'] >= 0, '風圧_S'] =  1.2 / 2 * c_in  * df['風速_S'] ** 2
    df.loc[df['風速_S'] < 0,  '風圧_S'] = -1.2 / 2 * c_out * df['風速_S'] ** 2
    df.loc[df['風速_W'] >= 0, '風圧_W'] =  1.2 / 2 * c_in  * df['風速_W'] ** 2
    df.loc[df['風速_W'] < 0,  '風圧_W'] = -1.2 / 2 * c_out * df['風速_W'] ** 2
    df.loc[df['風速_N'] >= 0, '風圧_N'] =  1.2 / 2 * c_in  * df['風速_N'] ** 2
    df.loc[df['風速_N'] < 0,  '風圧_N'] = -1.2 / 2 * c_out * df['風速_N'] ** 2
    df['風圧_H']                        =  1.2 / 2 * c_horizontal * (s ** 2)

    wind_pressure = {
        'E': df['風圧_E'],
        'S': df['風圧_S'],
        'W': df['風圧_W'],
        'N': df['風圧_N'],
        'H': df['風圧_H'],
    }

    return df, wind_pressure

import numpy as np
import pandas as pd

def make_wind(d, s, c_in = 0.7, c_out = -0.55, c_horizontal = -0.90):
    """風向・風速から各方位の風圧を算出する
    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数
    戻り値:
      (DataFrame, dict[str, Series]) 風向・風速などの中間列と、方位別風圧（E/S/W/N/H）
    """
    df = pd.DataFrame(index = d.index) 

    df['風速_E'] =   np.sin(np.radians(d * 22.5)) * s
    df['風速_S'] = - np.cos(np.radians(d * 22.5)) * s
    df['風速_W'] = - np.sin(np.radians(d * 22.5)) * s
    df['風速_N'] =   np.cos(np.radians(d * 22.5)) * s

    df.loc[df['風速_E'] >= 0, '風圧_E'] =  1.2 / 2 * c_in  * df['風速_E'] ** 2
    df.loc[df['風速_E'] < 0,  '風圧_E'] = -1.2 / 2 * c_out * df['風速_E'] ** 2
    df.loc[df['風速_S'] >= 0, '風圧_S'] =  1.2 / 2 * c_in  * df['風速_S'] ** 2
    df.loc[df['風速_S'] < 0,  '風圧_S'] = -1.2 / 2 * c_out * df['風速_S'] ** 2
    df.loc[df['風速_W'] >= 0, '風圧_W'] =  1.2 / 2 * c_in  * df['風速_W'] ** 2
    df.loc[df['風速_W'] < 0,  '風圧_W'] = -1.2 / 2 * c_out * df['風速_W'] ** 2
    df.loc[df['風速_N'] >= 0, '風圧_N'] =  1.2 / 2 * c_in  * df['風速_N'] ** 2
    df.loc[df['風速_N'] < 0,  '風圧_N'] = -1.2 / 2 * c_out * df['風速_N'] ** 2
    df['風圧_H']                        =  1.2 / 2 * c_horizontal * (s ** 2)

    wind_pressure = {
        'E': df['風圧_E'],
        'S': df['風圧_S'],
        'W': df['風圧_W'],
        'N': df['風圧_N'],
        'H': df['風圧_H'],
    }

    return df, wind_pressure
import pandas as pd
import numpy as np

def make_wind(d, s, c_in = 0.7, c_out = -0.55, c_horizontal = -0.90):
    """風向・風速から各方位の風圧を算出する
    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数
    戻り値:
      (DataFrame, dict[str, Series]) 風向・風速などの中間列と、方位別風圧（E/S/W/N/H）
    """
    df = pd.DataFrame(index = d.index)

    df['風速_E'] =   np.sin(np.radians(d * 22.5)) * s
    df['風速_S'] = - np.cos(np.radians(d * 22.5)) * s
    df['風速_W'] = - np.sin(np.radians(d * 22.5)) * s
    df['風速_N'] =   np.cos(np.radians(d * 22.5)) * s

    df.loc[df['風速_E'] >= 0, '風圧_E'] =  1.2 / 2 * c_in  * df['風速_E'] ** 2
    df.loc[df['風速_E'] < 0,  '風圧_E'] = -1.2 / 2 * c_out * df['風速_E'] ** 2
    df.loc[df['風速_S'] >= 0, '風圧_S'] =  1.2 / 2 * c_in  * df['風速_S'] ** 2
    df.loc[df['風速_S'] < 0,  '風圧_S'] = -1.2 / 2 * c_out * df['風速_S'] ** 2
    df.loc[df['風速_W'] >= 0, '風圧_W'] =  1.2 / 2 * c_in  * df['風速_W'] ** 2
    df.loc[df['風速_W'] < 0,  '風圧_W'] = -1.2 / 2 * c_out * df['風速_W'] ** 2
    df.loc[df['風速_N'] >= 0, '風圧_N'] =  1.2 / 2 * c_in  * df['風速_N'] ** 2
    df.loc[df['風速_N'] < 0,  '風圧_N'] = -1.2 / 2 * c_out * df['風速_N'] ** 2
    df['風圧_H']                        =  1.2 / 2 * c_horizontal * (s ** 2)

    wind_pressure = {
        'E': df['風圧_E'],
        'S': df['風圧_S'],
        'W': df['風圧_W'],
        'N': df['風圧_N'],
        'H': df['風圧_H'],
    }

    return df, wind_pressure

__all__ = ["make_wind"]

import pandas as pd
import numpy as np


def make_wind(d, s, c_in = 0.7, c_out = -0.55, c_horizontal = -0.90):
    """風向・風速から各方位の風圧を算出する
    引数:
      d: 風向カテゴリ（0:無風, 1:NNE, ..., 16:N）Series
      s: 風速 [m/s] Series
      c_in, c_out, c_horizontal: 風圧係数
    戻り値:
      (DataFrame, dict[str, Series]) 風向・風速などの中間列と、方位別風圧（E/S/W/N/H）
    """
    df = pd.DataFrame(index = d.index) 

    df['風速_E'] =   np.sin(np.radians(d * 22.5)) * s
    df['風速_S'] = - np.cos(np.radians(d * 22.5)) * s
    df['風速_W'] = - np.sin(np.radians(d * 22.5)) * s
    df['風速_N'] =   np.cos(np.radians(d * 22.5)) * s

    df.loc[df['風速_E'] >= 0, '風圧_E'] =  1.2 / 2 * c_in  * df['風速_E'] ** 2
    df.loc[df['風速_E'] < 0,  '風圧_E'] = -1.2 / 2 * c_out * df['風速_E'] ** 2
    df.loc[df['風速_S'] >= 0, '風圧_S'] =  1.2 / 2 * c_in  * df['風速_S'] ** 2
    df.loc[df['風速_S'] < 0,  '風圧_S'] = -1.2 / 2 * c_out * df['風速_S'] ** 2
    df.loc[df['風速_W'] >= 0, '風圧_W'] =  1.2 / 2 * c_in  * df['風速_W'] ** 2
    df.loc[df['風速_W'] < 0,  '風圧_W'] = -1.2 / 2 * c_out * df['風速_W'] ** 2
    df.loc[df['風速_N'] >= 0, '風圧_N'] =  1.2 / 2 * c_in  * df['風速_N'] ** 2
    df.loc[df['風速_N'] < 0,  '風圧_N'] = -1.2 / 2 * c_out * df['風速_N'] ** 2
    df['風圧_H']                        =  1.2 / 2 * c_horizontal * (s ** 2)

    wind_pressure = {
        'E': df['風圧_E'],
        'S': df['風圧_S'],
        'W': df['風圧_W'],
        'N': df['風圧_N'],
        'H': df['風圧_H'],
    }
    return df, wind_pressure



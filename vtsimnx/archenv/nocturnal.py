import pandas as pd
import numpy as np

from .archenv import e, T, MJ_to_Wh, Sigma


# 夜間放射 MJ/m2
rn = lambda t, h: (94.21 + 39.06 * np.sqrt(e(t, h) / 100) \
                   - 0.85 * Sigma * np.power(T(t), 4)) * 4.187 / 1000


def make_nocturnal(**kwargs):
    """夜間放射量を算出する
    指定方法:
      - t, h を与える: 気温・相対湿度から推算
      - 夜間放射量（'夜間放射量' or 'n_r'）を与える: その値を使用
    戻り値:
      (DataFrame) 夜間放射量
    """
    if '外気温' in kwargs:
        t = kwargs['外気温']
        h = kwargs['外気相対湿度']
        df = pd.DataFrame(index = t.index)
        df['夜間放射量'] = MJ_to_Wh(rn(t, h))
    elif '夜間放射量' in kwargs:
        n_r = kwargs['夜間放射量']
        df = pd.DataFrame(index = n_r.index)
        df['夜間放射量'] = n_r
    elif 'n_r' in kwargs:
        n_r = kwargs['n_r']
        df = pd.DataFrame(index = n_r.index)
        df['夜間放射量'] = n_r
    else:
        raise Exception('ERROR: 外気温 t がありません。夜間放射量 n_r もありません。')

    return df

import pandas as pd
import numpy as np

from .archenv import T, e, Sigma, MJ_to_Wh

# 夜間放射 MJ/m2
rn = lambda t, h: (94.21 + 39.06 * np.sqrt(e(t, h) / 100) \
                   - 0.85 * Sigma * np.power(T(t), 4)) * 4.187 / 1000

def make_nocturnal(**kwargs):
    """夜間放射量を算出する
    指定方法:
      - t, h を与える: 気温・相対湿度から推算
      - n_r を与える: 夜間放射量を直接使用
    戻り値:
      (DataFrame) 夜間放射量
    """
    if('外気温' in kwargs):
        t = kwargs['外気温']
        h = kwargs['外気相対湿度']
        df = pd.DataFrame(index = t.index)
        df['夜間放射量'] = MJ_to_Wh(rn(t, h))
    elif('n_r' in kwargs):
        n_r = kwargs['夜間放射量']
        df = pd.DataFrame(index = n_r.index)
        df['夜間放射量'] = n_r
    else:
        raise Exception('ERROR: 外気温 t がありません。夜間放射量 n_r もありません。')

    return df

__all__ = ["rn", "make_nocturnal"]



import pandas as pd
import numpy as np
import datetime

from .archenv import Wh_to_MJ, MJ_to_Wh, _az_deg_from_sin_cos, _alt_deg_from_sin
from .archenv import Solar_I


# 直散分離（Erbs）
Kt = lambda IG, alt: IG / (Wh_to_MJ(Solar_I) * np.sin(np.radians(alt)))  # 晴天指数


def _as_array(x):
    """pandas Series を含む配列状入力を numpy 配列に正規化する"""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def Id(IG, kt):
    """水平面拡散日射量の推定（Erbs 法）
    IG: 水平面全天日射量 [Wh/m2] 配列
    kt: 晴天指数
    """
    IG_arr = _as_array(IG)
    kt_arr = _as_array(kt)
    s_Id = np.zeros(len(kt_arr))
    for i, k in enumerate(kt_arr):
        if   k <= 0.22:                 s_Id[i] = IG_arr[i] * (1 - 0.09 * k)
        elif (0.22 < k) & (k <= 0.80):  s_Id[i] = IG_arr[i] * (0.9511 -  0.1604 * k \
                                                                  +  4.388  * np.power(k, 2) \
                                                                  - 16.638  * np.power(k, 3) \
                                                                  + 12.336  * np.power(k, 4))
        elif 0.80 < k:                  s_Id[i] = 0.365 * IG_arr[i]
    return s_Id


def Ib(IG, Id, alt, min_alt_deg: float = 0.0):
    """法線面直達日射量の推定
    IG: 水平面全天日射量 [Wh/m2]
    Id: 水平面拡散日射量 [Wh/m2]
    alt: 太陽高度 [deg]
    min_alt_deg: 太陽高度がこの値未満のとき直達を 0 扱い（スパイク抑制用）
    """
    IG_arr  = _as_array(IG)
    Id_arr  = _as_array(Id)
    alt_arr = _as_array(alt)
    s_Ib = np.zeros(len(Id_arr))
    for i, idv in enumerate(Id_arr):
        alt_i = float(alt_arr[i])
        if alt_i <= float(min_alt_deg):
            s_Ib[i] = 0.0
            continue
        s = np.sin(np.radians(alt_i))
        # 数値的に極小な sin での発散を避ける
        if s <= 0.0:
            s_Ib[i] = 0.0
            continue

        s_Ib[i] = (IG_arr[i] - idv) / s
        # 低高度での異常値対策（既存互換）
        if (alt_i < 10.0) & (s_Ib[i] > IG_arr[i]):
            s_Ib[i] = IG_arr[i]
    return s_Ib


# 太陽位置の計算に用いる基本式
# 太陽の赤緯 δ [deg]（年周の余弦和による近似）
delta_d = lambda N: (180 / np.pi) * (0.006322 \
                                     - 0.405748 * np.cos(2 * np.pi * N / 366 + 0.153231) \
                                     - 0.005880 * np.cos(4 * np.pi * N / 366 + 0.207099) \
                                     - 0.003233 * np.cos(6 * np.pi * N / 366 + 0.620129))

# 太陽の均時差 ed [h]（平均太陽時と真太陽時の差）
e_d     = lambda N: -0.000279 + 0.122772 * np.cos(2 * np.pi * N / 366 + 1.498311) \
                              - 0.165458 * np.cos(4 * np.pi * N / 366 - 1.261546) \
                              - 0.005354 * np.cos(6 * np.pi * N / 366 - 1.1571)

# 太陽の時角 T_d_t [deg]（正午=0°、午前は負、午後は正）
T_d_t   = lambda H, ed, L       : (H  + ed - 12.0) * 15.0 + (L - 135.0)

# 角度[deg]の正弦・余弦
sin     = lambda v              : np.sin(np.radians(v))
cos     = lambda v              : np.cos(np.radians(v))

# 太陽高度 hs の正弦: sin(hs) = sin(lat)·sin(δ) + cos(lat)·cos(δ)·cos(時角)
sin_hs  = lambda L, dd, tdt     : sin(L) * sin(dd) + cos(L) * cos(dd) * cos(tdt)
# 太陽方位角 AZs の正弦/余弦（象限判定は arctan2 で別途実施）
sin_AZs = lambda dd, tdt, c_h   : cos(dd) * sin(tdt) / c_h
cos_AZs = lambda s_h, L, dd, c_h: (s_h * sin(L) - sin(dd)) / (c_h * cos(L))


def sun_loc(idx, lat = 36.00, lon = 140.00, td = -0.5):
    """太陽位置を簡易式で算出（赤緯/均時差/時角から）
    idx: DatetimeIndex
    lat, lon: 緯度・経度 [deg]
    td: ローカル時刻微調整 [h]
    """
    df = pd.DataFrame(index=idx)
    # 元日からの通し日数（正午寄せ）
    df['元日からの通し日数 N'] = [
        (i - datetime.datetime(i.year, 1, 1)).days + 1.5
        for i in idx
    ]
    # 小数時間（時＋分/60）に微調整 td を加える
    df['時刻 H'] = (
        idx.strftime("%H").astype('float64')
        + idx.strftime("%M").astype('float64') / 60
        + td
    )
    # 太陽の基本角
    df['太陽の赤緯 delta_d'] = delta_d(df['元日からの通し日数 N'])
    df['太陽の均時差 e_d']   = e_d(df['元日からの通し日数 N'])
    df['太陽の時角 T_d_t']   = T_d_t(df['時刻 H'], df['太陽の均時差 e_d'], lon)
    # 太陽高度
    df['太陽高度の正弦 sin_hs'] = sin_hs(
        lat, df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t']
    )
    # 数値誤差で sin_hs が [-1, 1] を僅かに逸脱すると sqrt が NaN になったり、
    # cos_hs ≈ 0（天頂近傍）で方位角の sin/cos 計算が不安定になる。
    # ここでクリップ＆安全な cos_hs を作る。
    df['太陽高度の正弦 sin_hs'] = np.clip(df['太陽高度の正弦 sin_hs'], -1.0, 1.0)
    df['太陽高度の余弦 cos_hs'] = np.sqrt(
        np.clip(1 - np.power(df['太陽高度の正弦 sin_hs'], 2), 0.0, 1.0)
    )
    df['太陽高度 hs'] = _alt_deg_from_sin(df['太陽高度の正弦 sin_hs'])

    # 太陽方位角
    # cos_hs が極小だと分母が不安定になるので、近傍は方位角が定義できない前提で丸め込む
    eps = 1e-12
    safe_cos_hs = np.where(df['太陽高度の余弦 cos_hs'] < eps, np.nan, df['太陽高度の余弦 cos_hs'])
    df['太陽方位角の正弦 sin_AZs'] = sin_AZs(
        df['太陽の赤緯 delta_d'], df['太陽の時角 T_d_t'], safe_cos_hs
    )
    df['太陽方位角の余弦 cos_AZs'] = cos_AZs(
        df['太陽高度の正弦 sin_hs'], lat, df['太陽の赤緯 delta_d'], safe_cos_hs
    )
    # 数値誤差で [-1, 1] を僅かに逸脱することがあるためクリップ
    df['太陽方位角の正弦 sin_AZs'] = np.clip(df['太陽方位角の正弦 sin_AZs'], -1.0, 1.0)
    df['太陽方位角の余弦 cos_AZs'] = np.clip(df['太陽方位角の余弦 cos_AZs'], -1.0, 1.0)

    df['太陽方位角 AZs'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs']
    )

    return df


def astro_sun_loc(idx, lat = '36 00 00.00', lon = '140 00 00.00', td = -0.5):
    """astropy を用いた太陽位置の高精度計算
    lat, lon は DMS 表記の文字列を想定（例: '36 00 00.00'）
    戻り値: 仰角/方位角の sin, cos と角度 [deg]
    """
    # IERS（地球回転パラメータ）を自動取得して精度劣化の警告を避ける
    # オフライン等で取得できない場合もあるため、例外は握りつぶして続行する
    try:
        from astropy.utils import iers

        iers.conf.auto_download = True
        iers.conf.iers_auto_url = "https://datacenter.iers.org/data/9/finals2000A.all"
    except Exception:
        pass

    import astropy.time
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.coordinates import AltAz
    from astropy.coordinates import EarthLocation

    # 互換: float（deg）でも渡せるようにする
    # EarthLocation は unit 付きの文字列（例: "36d"）を受け付ける
    if isinstance(lat, (int, float)):
        lat = f"{float(lat)}d"
    if isinstance(lon, (int, float)):
        lon = f"{float(lon)}d"

    loc = EarthLocation(lat = lat, lon = lon)
    time = astropy.time.Time(idx) + (-9 + td ) * u.hour
    sun = get_sun(time).transform_to(AltAz(obstime = time, location = loc))

    df = pd.DataFrame(index = idx)

    # 太陽高度（仰角）
    df['太陽高度の正弦 sin_alt'] = np.array([np.sin(s.alt) for s in sun]).astype('float64')
    df['太陽高度の余弦 cos_alt'] = np.array([np.cos(s.alt) for s in sun]).astype('float64')
    df['太陽高度 alt'] = np.degrees(
        np.arcsin(df['太陽高度の正弦 sin_alt'])
    )

    # 太陽方位角
    df['太陽方位角の正弦 sin_az'] = np.array([np.sin(s.az) for s in sun]).astype('float64')
    df['太陽方位角の余弦 cos_az'] = np.array([np.cos(s.az) for s in sun]).astype('float64')

    df['太陽方位角 az'] = _az_deg_from_sin_cos(
        df['太陽方位角の正弦 sin_az'], df['太陽方位角の余弦 cos_az']
    )

    return df


def sep_direct_diffuse(s_ig, s_hs, min_sun_alt_deg: float = 0.0):
    """全天日射量と太陽高度から直散分離（Erbs）を行い Kt/Id/Ib を返す"""
    df = pd.concat([s_ig, s_hs], axis = 1)
    df.columns = ['水平面全天日射量', '太陽高度']
    df['晴天指数 Kt'] = Kt(Wh_to_MJ(df['水平面全天日射量']), df['太陽高度'])
    df['水平面拡散日射量 Id'] = MJ_to_Wh(Id(Wh_to_MJ(df['水平面全天日射量']), df['晴天指数 Kt']))
    df['法線面直達日射量 Ib'] = MJ_to_Wh(
        Ib(
            Wh_to_MJ(df['水平面全天日射量']),
            Wh_to_MJ(df['水平面拡散日射量 Id']),
            df['太陽高度'],
            min_alt_deg=min_sun_alt_deg,
        )
    )
    return df


def eta(c):
    """角度補正の近似多項式
    P(c) = 2.3920·c − 3.8636·c^3 + 3.7568·c^5 − 1.3968·c^7
    Horner 形で評価して数値安定性と可読性を両立。
    """
    c2 = c * c
    return c * (2.3920 + c2 * (-3.8636 + c2 * (3.7568 + c2 * (-1.3968))))


def direc_solar(s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs):
    """直達・拡散・反射から方位別日射量（壁/ガラス/水平）を算出する"""
    df = pd.concat(
        [s_ib, s_id, s_sin_hs, s_cos_hs, s_hs, s_sin_AZs, s_cos_AZs, s_AZs],
        axis = 1
    )
    df.columns = [
        '法線面直達日射量 Ib', '水平面拡散日射量 Id',
        '太陽高度の正弦 sin_hs', '太陽高度の余弦 cos_hs', '太陽高度 hs',
        '太陽方位角の正弦 sin_AZs', '太陽方位角の余弦 cos_AZs', '太陽方位角 AZs'
    ]

    cond = df['太陽高度 hs'] > 0
    az = df['太陽方位角 AZs']
    cos_hs = df['太陽高度の余弦 cos_hs']
    sin_hs = df['太陽高度の正弦 sin_hs']
    sin_az = df['太陽方位角の正弦 sin_AZs']
    cos_az = df['太陽方位角の余弦 cos_AZs']
    Ib_col = df['法線面直達日射量 Ib']

    df.loc[cond & (-180 < az) & (az < 0),   '直達日射量の東面成分 Ib_E'] = (
        -1 * Ib_col * cos_hs * sin_az
    )  # 東
    df.loc[cond & (-90  < az) & (az < 90),  '直達日射量の南面成分 Ib_S'] = (
           Ib_col * cos_hs * cos_az
    )  # 南
    df.loc[cond & (0    < az) & (az < 180), '直達日射量の西面成分 Ib_W'] = (
           Ib_col * cos_hs * sin_az
    )  # 西
    df.loc[cond & (-180 < az) & (az < -90), '直達日射量の北面成分 Ib_N'] = (
        -1 * Ib_col * cos_hs * cos_az
    )  # 北
    df.loc[cond & (  90 < az) & (az < 180), '直達日射量の北面成分 Ib_N'] = (
        -1 * Ib_col * cos_hs * cos_az
    )  # 北
    df.loc[cond, '直達日射量の水平面成分 Ib_H'] = (
        Ib_col * sin_hs
    )
    df.loc[cond, '水平面拡散日射量の反射成分 Id_R'] = (
        (df['水平面拡散日射量 Id'] + Ib_col) * sin_hs * 0.5 * 0.1
    )
    df['水平面拡散日射量の拡散成分 Id_D'] = df['水平面拡散日射量 Id'] * 0.5

    df = df.fillna(0)

    df['直達日射量の東面成分（ガラス） Ib_E_g'] = (
        df['直達日射量の東面成分 Ib_E'] * eta(-1 * cos_hs * sin_az)
    )  # 東
    df['直達日射量の南面成分（ガラス） Ib_S_g'] = (
        df['直達日射量の南面成分 Ib_S'] * eta(      cos_hs * cos_az)
    )  # 南
    df['直達日射量の西面成分（ガラス） Ib_W_g'] = (
        df['直達日射量の西面成分 Ib_W'] * eta(      cos_hs * sin_az)
    )  # 西
    df['直達日射量の北面成分（ガラス） Ib_N_g'] = (
        df['直達日射量の北面成分 Ib_N'] * eta(-1 * cos_hs * cos_az)
    )  # 北
    df['直達日射量の水平面成分（ガラス） Ib_H_g'] = (
        df['直達日射量の水平面成分 Ib_H'] * eta(sin_hs)
    )  # 水平
    df['水平面拡散日射量の反射成分（ガラス） Id_R_g'] = (
        df['水平面拡散日射量の反射成分 Id_R'] * 0.808
    )  # 透過率
    df['水平面拡散日射量の拡散成分（ガラス） Id_D_g'] = (
        df['水平面拡散日射量の拡散成分 Id_D'] * 0.808
    )  # 透過率

    df['日射熱取得量（東面）'] = (
        df['直達日射量の東面成分 Ib_E']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（南面）'] = (
        df['直達日射量の南面成分 Ib_S']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（西面）'] = (
        df['直達日射量の西面成分 Ib_W']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（北面）'] = (
        df['直達日射量の北面成分 Ib_N']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（水平面）'] = (
        df['直達日射量の水平面成分 Ib_H']
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )
    df['日射熱取得量（拡散）'] = (
        + df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )

    df['日射熱取得量（東面ガラス）'] = (
        df['直達日射量の東面成分（ガラス） Ib_E_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（南面ガラス）'] = (
        df['直達日射量の南面成分（ガラス） Ib_S_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（西面ガラス）'] = (
        df['直達日射量の西面成分（ガラス） Ib_W_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（北面ガラス）'] = (
        df['直達日射量の北面成分（ガラス） Ib_N_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（水平面ガラス）'] = (
        df['直達日射量の水平面成分（ガラス） Ib_H_g']
        + df['水平面拡散日射量の拡散成分（ガラス） Id_D_g']
        + df['水平面拡散日射量の反射成分（ガラス） Id_R_g']
    )
    df['日射熱取得量（拡散）'] = (
        df['水平面拡散日射量の拡散成分 Id_D']
        + df['水平面拡散日射量の反射成分 Id_R']
    )

    return df


def make_solar(
    *args,
    use_astro: bool = False,
    time_alignment: str = "center",
    timestamp_ref: str = "start",
    min_sun_alt_deg: float = 0.0,
    **kwargs,
):
    """方位別日射量の総合算出
    指定方法:
      - 全天日射量を与えると内部で直散分離 → 方位別
      - 法線面直達日射量と水平面全天日射量を与えると Id=IG−Ib·sin(hs) で復元 → 方位別
      - 法線面直達日射量と水平面拡散日射量を与えるとそのまま方位別へ
    戻り値:
      (DataFrame) 中間列と方位別日射（壁/ガラス/水平）

    use_astro=True の場合は astropy を用いた高精度太陽位置（astro_sun_loc）を使用する。
    （astropy が未インストールの場合は ImportError になります）

    time_alignment / timestamp_ref:
      - time_alignment="center": タイムステップ区間の中央時刻で評価（td=±Δt/2）
      - time_alignment="timestamp": インデックス時刻そのもの（td=0）
      - timestamp_ref="start": インデックスが区間開始を表す
      - timestamp_ref="end":   インデックスが区間終了を表す（例: 00:00 が 23:00-00:00 の値）
    """
    if time_alignment not in ("center", "timestamp"):
        raise ValueError(f"time_alignment must be 'center' or 'timestamp', got {time_alignment!r}")
    if timestamp_ref not in ("start", "end"):
        raise ValueError(f"timestamp_ref must be 'start' or 'end', got {timestamp_ref!r}")
    if not (0.0 <= float(min_sun_alt_deg) <= 90.0):
        raise ValueError(f"min_sun_alt_deg must be in [0, 90], got {min_sun_alt_deg!r}")

    def _auto_td_from_index(idx: pd.DatetimeIndex) -> float:
        if time_alignment == "timestamp":
            return 0.0
        # center
        if len(idx) < 2:
            return 0.0
        delta_h = (idx[1] - idx[0]).total_seconds() / 3600.0
        # indexが区間開始なら +Δt/2、区間終了なら -Δt/2
        sgn = 1.0 if timestamp_ref == "start" else -1.0
        return sgn * delta_h / 2.0

    # 互換対応: 先頭位置引数で DataFrame/Series を受けた場合に自動でキーへマッピングする
    if args:
        first = args[0]
        if isinstance(first, pd.Series):
            # 旧API: make_solar(series_IG, lat=..., lon=...)
            kwargs.setdefault('全天日射量', first)
        elif isinstance(first, pd.DataFrame):
            df0 = first
            # 列名の別名に対応
            ig_col_candidates = ['全天日射量', '水平面全天日射量', 'IG']
            # HASP などでは「直達日射量」という列名で法線面直達（DNI 相当）が入ってくる想定
            ib_col_candidates = ['法線面直達日射量', '直達日射量', 'Ib']
            id_col_candidates = ['水平面拡散日射量', 'Id']
            ig_col = next((c for c in ig_col_candidates if c in df0.columns), None)
            ib_col = next((c for c in ib_col_candidates if c in df0.columns), None)
            id_col = next((c for c in id_col_candidates if c in df0.columns), None)
            # 優先順位:
            #  - IG + Ib が揃っているなら、それを使って Id を復元できる（今回の不具合の主因）
            #  - 次に Ib + Id
            #  - 最後に IG のみ（Erbs 直散分離）
            if ig_col is not None and ib_col is not None:
                kwargs.setdefault('全天日射量', df0[ig_col])
                kwargs.setdefault('法線面直達日射量', df0[ib_col])
            elif ib_col is not None and id_col is not None:
                kwargs.setdefault('法線面直達日射量', df0[ib_col])
                kwargs.setdefault('水平面拡散日射量', df0[id_col])
            elif ig_col is not None:
                kwargs.setdefault('全天日射量', df0[ig_col])
            # それ以外は kwargs のみで続行

    # 互換: キー名揺れをここで正規化（kwargs で直接渡されたケース）
    if '水平面全天日射量' in kwargs and '全天日射量' not in kwargs:
        kwargs['全天日射量'] = kwargs['水平面全天日射量']
    if '直達日射量' in kwargs and '法線面直達日射量' not in kwargs:
        kwargs['法線面直達日射量'] = kwargs['直達日射量']

    lat = kwargs['緯度'] if '緯度' in kwargs else 36.00
    lon = kwargs['経度'] if '経度' in kwargs else 140.00

    # ケースA: 法線面直達 + 水平面全天 が与えられた場合（直散分離はせず、Id を復元する）
    #   IG = Ib*sin(hs) + Id  →  Id = IG - Ib*sin(hs)
    if '全天日射量' in kwargs and '法線面直達日射量' in kwargs and '水平面拡散日射量' not in kwargs:
        s_ig = kwargs['全天日射量']
        s_ib_in = kwargs['法線面直達日射量']

        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            td = _auto_td_from_index(s_ig.index)

        # 太陽位置
        if use_astro:
            df_a = astro_sun_loc(s_ig.index, lat=lat, lon=lon, td=td)
            az = df_a["太陽方位角 az"]
            AZs = ((az - 180.0 + 180.0) % 360.0) - 180.0

            df_sun = pd.DataFrame(index=df_a.index)
            df_sun["太陽高度の正弦 sin_hs"] = df_a["太陽高度の正弦 sin_alt"]
            df_sun["太陽高度の余弦 cos_hs"] = df_a["太陽高度の余弦 cos_alt"]
            df_sun["太陽高度 hs"] = df_a["太陽高度 alt"]
            df_sun["太陽方位角 AZs"] = AZs
            df_sun["太陽方位角の正弦 sin_AZs"] = np.sin(np.radians(AZs))
            df_sun["太陽方位角の余弦 cos_AZs"] = np.cos(np.radians(AZs))
        else:
            df_sun = sun_loc(s_ig.index, lat=lat, lon=lon, td=td)

        # Id 復元（夜間は 0、また Ib*sin(hs) が IG を超える場合は Ib を上限で丸める）
        sin_hs = df_sun["太陽高度の正弦 sin_hs"].astype("float64")
        hs = df_sun["太陽高度 hs"].astype("float64")
        ig = s_ig.astype("float64")
        ib = s_ib_in.astype("float64")

        day = (hs > float(min_sun_alt_deg)) & (sin_hs > 0)
        ib_used = pd.Series(0.0, index=s_ig.index)
        id_used = pd.Series(0.0, index=s_ig.index)
        if day.any():
            ib_h = ib[day] * sin_hs[day]
            # 直達水平成分が IG を超える場合、Ib を IG/sin(hs) に制限（Id=0 になる）
            ib_cap = ig[day] / sin_hs[day]
            ib_eff = np.where(ib_h > ig[day], ib_cap, ib[day])
            ib_used.loc[day] = ib_eff
            id_eff = ig[day] - ib_used.loc[day] * sin_hs[day]
            id_used.loc[day] = np.maximum(id_eff, 0.0)

        # 出力（IG も残しておく）
        df = pd.concat(
            [
                ig.rename("水平面全天日射量"),
                ib_used.rename("法線面直達日射量 Ib"),
                id_used.rename("水平面拡散日射量 Id"),
                df_sun,
            ],
            axis=1,
        )
        df = direc_solar(
            df["法線面直達日射量 Ib"],
            df["水平面拡散日射量 Id"],
            df["太陽高度の正弦 sin_hs"],
            df["太陽高度の余弦 cos_hs"],
            df["太陽高度 hs"],
            df["太陽方位角の正弦 sin_AZs"],
            df["太陽方位角の余弦 cos_AZs"],
            df["太陽方位角 AZs"],
        )
        # direc_solar の df に IG を戻す（列順は気にしない）
        df = pd.concat([ig.rename("水平面全天日射量"), df], axis=1)

    # ケースB: 水平面全天のみ（Erbs 直散分離）
    elif '全天日射量' in kwargs:
        s_ig = kwargs['全天日射量']
        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            td = _auto_td_from_index(s_ig.index)
        if use_astro:
            # astropy の方位角は 0°=北, 90°=東（一般的な測地系）
            # 本コードの方位角 AZs は 0°=南, -90°=東, +90°=西, ±180°=北 を前提に
            # direc_solar 内で象限判定しているため、ここで座標系を変換する。
            df_a = astro_sun_loc(s_ig.index, lat=lat, lon=lon, td=td)

            # astro az(=0北) -> 本コードAZs(=0南)
            # wrap を [-180, 180] に揃える
            az = df_a["太陽方位角 az"]
            AZs = ((az - 180.0 + 180.0) % 360.0) - 180.0

            df_sun = pd.DataFrame(index=df_a.index)
            df_sun["太陽高度の正弦 sin_hs"] = df_a["太陽高度の正弦 sin_alt"]
            df_sun["太陽高度の余弦 cos_hs"] = df_a["太陽高度の余弦 cos_alt"]
            df_sun["太陽高度 hs"] = df_a["太陽高度 alt"]
            df_sun["太陽方位角 AZs"] = AZs
            df_sun["太陽方位角の正弦 sin_AZs"] = np.sin(np.radians(AZs))
            df_sun["太陽方位角の余弦 cos_AZs"] = np.cos(np.radians(AZs))
        else:
            df_sun = sun_loc(s_ig.index, lat=lat, lon=lon, td=td)

        df = pd.concat([s_ig, df_sun], axis=1)
        df = pd.concat(
            [df, sep_direct_diffuse(s_ig, df['太陽高度 hs'], min_sun_alt_deg=min_sun_alt_deg)],
            axis = 1
        )  # 直散分離結果
        df = direc_solar(
            df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],              # 方位別日射量
            df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
            df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs']
        )
    else:
        if '法線面直達日射量' in kwargs:
            s_ib = kwargs['法線面直達日射量']
        else:
            raise Exception('ERROR: 法線面直達日射量 s_ib がありません。')
        if '水平面拡散日射量' in kwargs:
            s_id = kwargs['水平面拡散日射量']
        else:
            raise Exception('ERROR: 水平面拡散日射量 s_id がありません。')

        # 入力 Series の name は任意なので、以降の参照に合わせてここで正規化する
        s_ib = s_ib.rename('法線面直達日射量 Ib')
        s_id = s_id.rename('水平面拡散日射量 Id')
        if '時刻調整' in kwargs:
            td = kwargs['時刻調整']
        else:
            td = _auto_td_from_index(s_ib.index)
        if use_astro:
            df_a = astro_sun_loc(s_ib.index, lat=lat, lon=lon, td=td)
            az = df_a["太陽方位角 az"]
            AZs = ((az - 180.0 + 180.0) % 360.0) - 180.0

            df_sun = pd.DataFrame(index=df_a.index)
            df_sun["太陽高度の正弦 sin_hs"] = df_a["太陽高度の正弦 sin_alt"]
            df_sun["太陽高度の余弦 cos_hs"] = df_a["太陽高度の余弦 cos_alt"]
            df_sun["太陽高度 hs"] = df_a["太陽高度 alt"]
            df_sun["太陽方位角 AZs"] = AZs
            df_sun["太陽方位角の正弦 sin_AZs"] = np.sin(np.radians(AZs))
            df_sun["太陽方位角の余弦 cos_AZs"] = np.cos(np.radians(AZs))
        else:
            df_sun = sun_loc(s_ib.index, lat=lat, lon=lon, td=td)

        df = pd.concat([s_ib, s_id, df_sun], axis=1)
        df = direc_solar(
            df['法線面直達日射量 Ib'], df['水平面拡散日射量 Id'],
            df['太陽高度の正弦 sin_hs'], df['太陽高度の余弦 cos_hs'], df['太陽高度 hs'],
            df['太陽方位角の正弦 sin_AZs'], df['太陽方位角の余弦 cos_AZs'], df['太陽方位角 AZs']
        )

    return df

__all__ = [
    "Kt", "Id", "Ib",
    "delta_d", "e_d", "T_d_t",
    "sun_loc", "astro_sun_loc",
    "sep_direct_diffuse", "eta", "direc_solar", "make_solar",
]



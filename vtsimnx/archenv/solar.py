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


def _solar_gain_by_angles_from_solar_df(
    df_solar: pd.DataFrame,
    *,
    方位角: float,
    傾斜角: float,
    日射モード: str = "all",
    名前: str = "任意面",
    albedo: float = 0.1,
    glass_tau_diffuse: float = 0.808,
) -> pd.DataFrame:
    """
    必要な中間列（Ib/Id と太陽位置）を持つ DataFrame から、任意の面（方位角/傾斜角）の
    日射熱取得量（壁面/ガラス面）を計算する（内部ユーティリティ）。

    入力:
      df_solar: 内部計算用DataFrame（少なくとも以下の列が必要）
        - '法線面直達日射量 Ib'
        - '水平面拡散日射量 Id'
        - '太陽高度の正弦 sin_hs'
        - '太陽高度の余弦 cos_hs'
        - '太陽方位角 AZs' （0=南, -90=東, +90=西, ±180=北）

    角度の定義:
      - 方位角: 面の法線方位 [deg]（本コードのAZs座標系）
      - 傾斜角: 面の傾斜角 [deg]（0=水平上向き, 90=鉛直）

    出力:
      壁面: 直達 + 拡散 + 反射
      ガラス面: 直達(eta補正) + 拡散/反射(一定透過率)

    注:
      既存の E/S/W/N の結果との整合のため、拡散/反射は「垂直面(=0.5)」を基準に
      傾斜角に応じて view factor でスケールする簡易モデルを採用する。
    """
    if 日射モード not in ("all", "diffuse_only"):
        raise ValueError(f"日射モード must be 'all' or 'diffuse_only', got {日射モード!r}")
    required = [
        "法線面直達日射量 Ib",
        "水平面拡散日射量 Id",
        "太陽高度の正弦 sin_hs",
        "太陽高度の余弦 cos_hs",
        "太陽方位角 AZs",
    ]
    missing = [c for c in required if c not in df_solar.columns]
    if missing:
        raise KeyError(f"solar_gain_by_angles: df_solar に必要列がありません: {missing}")

    ib = df_solar["法線面直達日射量 Ib"].astype("float64")
    id_ = df_solar["水平面拡散日射量 Id"].astype("float64")
    sin_hs = df_solar["太陽高度の正弦 sin_hs"].astype("float64")
    cos_hs = df_solar["太陽高度の余弦 cos_hs"].astype("float64")
    azs = df_solar["太陽方位角 AZs"].astype("float64")

    beta = np.radians(float(傾斜角))
    gamma = float(方位角)

    # 入射角の余弦 cos(theta)
    # cosθ = sin(hs)*cosβ + cos(hs)*sinβ*cos(AZs-γ)
    cos_theta = (
        sin_hs * float(np.cos(beta))
        + cos_hs * float(np.sin(beta)) * np.cos(np.radians(azs - gamma))
    )
    cos_theta = np.maximum(cos_theta, 0.0)

    # 直達（面上）
    ib_surf = ib * cos_theta
    if 日射モード == "diffuse_only":
        ib_surf = ib_surf * 0.0

    # 拡散/反射（既存互換の簡易モデルを傾斜角でスケール）
    # 垂直面基準: Id_D = Id*0.5, Id_R = (Id+Ib)*sin(hs)*0.5*albedo
    # 傾斜角で view factor:
    #   F_sky   = (1+cosβ)/2, F_ground = (1-cosβ)/2
    f_sky = (1.0 + float(np.cos(beta))) / 2.0
    f_gnd = (1.0 - float(np.cos(beta))) / 2.0
    id_d = id_ * f_sky
    ib_for_ref = ib if 日射モード == "all" else (ib * 0.0)
    id_r = (id_ + ib_for_ref) * np.maximum(sin_hs, 0.0) * float(albedo) * f_gnd

    wall_gain = ib_surf + id_d + id_r

    # ガラス（直達はeta(cosθ)で角度補正、拡散/反射は一定透過率）
    ib_glass = ib_surf * eta(cos_theta)
    if 日射モード == "diffuse_only":
        ib_glass = ib_glass * 0.0
    id_d_g = id_d * float(glass_tau_diffuse)
    id_r_g = id_r * float(glass_tau_diffuse)
    glass_gain = ib_glass + id_d_g + id_r_g

    out = pd.DataFrame(index=df_solar.index)
    out[f"入射角cos({名前})"] = cos_theta
    out[f"直達日射量の{名前}面成分 Ib"] = ib_surf
    out[f"水平面拡散日射量の拡散成分（{名前}）"] = id_d
    out[f"水平面拡散日射量の反射成分（{名前}）"] = id_r
    out[f"日射熱取得量（{名前}）"] = wall_gain

    out[f"直達日射量の{名前}面成分（ガラス） Ib_g"] = ib_glass
    out[f"水平面拡散日射量の拡散成分（ガラス）（{名前}）"] = id_d_g
    out[f"水平面拡散日射量の反射成分（ガラス）（{名前}）"] = id_r_g
    out[f"日射熱取得量（{名前}ガラス）"] = glass_gain

    return out


def _solar_base(
    *,
    全天日射量: pd.Series | None,
    法線面直達日射量: pd.Series | None,
    水平面拡散日射量: pd.Series | None,
    緯度: float,
    経度: float,
    use_astro: bool,
    time_alignment: str,
    timestamp_ref: str,
    min_sun_alt_deg: float,
) -> pd.DataFrame:
    """
    太陽位置（sin/cos/角度）と、Ib/Id（法線面直達・水平面拡散）だけを作る最小ユーティリティ。
    """
    if time_alignment not in ("center", "timestamp"):
        raise ValueError(f"time_alignment must be 'center' or 'timestamp', got {time_alignment!r}")
    if timestamp_ref not in ("start", "end"):
        raise ValueError(f"timestamp_ref must be 'start' or 'end', got {timestamp_ref!r}")
    if not (0.0 <= float(min_sun_alt_deg) <= 90.0):
        raise ValueError(f"min_sun_alt_deg must be in [0, 90], got {min_sun_alt_deg!r}")

    # 入力の組み合わせチェック
    if 全天日射量 is None and not (法線面直達日射量 is not None and 水平面拡散日射量 is not None):
        raise TypeError("solar_gain_by_angles: 全天日射量 か、(法線面直達日射量 and 水平面拡散日射量) を指定してください。")

    idx = (全天日射量.index if 全天日射量 is not None else 法線面直達日射量.index)  # type: ignore[union-attr]

    def _auto_td_from_index(idx_: pd.DatetimeIndex) -> float:
        if time_alignment == "timestamp":
            return 0.0
        if len(idx_) < 2:
            return 0.0
        delta_h = (idx_[1] - idx_[0]).total_seconds() / 3600.0
        sgn = 1.0 if timestamp_ref == "start" else -1.0
        return sgn * delta_h / 2.0

    td = _auto_td_from_index(idx)

    # 太陽位置
    if use_astro:
        df_a = astro_sun_loc(idx, lat=緯度, lon=経度, td=td)
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
        df_sun = sun_loc(idx, lat=緯度, lon=経度, td=td)

    # Ib/Id の決定
    if 法線面直達日射量 is not None and 水平面拡散日射量 is not None:
        ib = 法線面直達日射量.astype("float64")
        id_ = 水平面拡散日射量.astype("float64")
        df_min = pd.DataFrame(index=idx)
        df_min["法線面直達日射量 Ib"] = ib
        df_min["水平面拡散日射量 Id"] = id_
        return pd.concat([df_min, df_sun], axis=1)

    if 全天日射量 is None:
        raise TypeError("solar_gain_by_angles: 全天日射量 がありません。")

    ig = 全天日射量.astype("float64")
    sin_hs = df_sun["太陽高度の正弦 sin_hs"].astype("float64")
    hs = df_sun["太陽高度 hs"].astype("float64")
    day = (hs > float(min_sun_alt_deg)) & (sin_hs > 0)

    # (A) IG + Ib で与えられた場合: Id = IG - Ib*sin(hs) を復元（Ibは上限で丸める）
    if 法線面直達日射量 is not None:
        ib_in = 法線面直達日射量.astype("float64")
        ib_used = pd.Series(0.0, index=idx)
        id_used = pd.Series(0.0, index=idx)
        if day.any():
            ib_h = ib_in[day] * sin_hs[day]
            ib_cap = ig[day] / sin_hs[day]
            ib_eff = np.where(ib_h > ig[day], ib_cap, ib_in[day])
            ib_used.loc[day] = ib_eff
            id_eff = ig[day] - ib_used.loc[day] * sin_hs[day]
            id_used.loc[day] = np.maximum(id_eff, 0.0)
        df_min = pd.DataFrame(index=idx)
        df_min["法線面直達日射量 Ib"] = ib_used
        df_min["水平面拡散日射量 Id"] = id_used
        df_min["水平面全天日射量"] = ig
        return pd.concat([df_min, df_sun], axis=1)

    # (B) IG のみ: Erbs 直散分離
    df_sep = sep_direct_diffuse(ig, df_sun["太陽高度 hs"], min_sun_alt_deg=min_sun_alt_deg)
    df_min = pd.DataFrame(index=idx)
    df_min["法線面直達日射量 Ib"] = df_sep["法線面直達日射量 Ib"]
    df_min["水平面拡散日射量 Id"] = df_sep["水平面拡散日射量 Id"]
    df_min["水平面全天日射量"] = ig
    return pd.concat([df_min, df_sun], axis=1)


def solar_gain_by_angles(
    *,
    方位角: float,
    傾斜角: float,
    緯度: float = 36.00,
    経度: float = 140.00,
    全天日射量: pd.Series | None = None,
    法線面直達日射量: pd.Series | None = None,
    水平面拡散日射量: pd.Series | None = None,
    名前: str = "任意面",
    use_astro: bool = False,
    time_alignment: str = "center",
    timestamp_ref: str = "start",
    min_sun_alt_deg: float = 0.0,
    日射モード: str = "all",
    albedo: float = 0.1,
    glass_tau_diffuse: float = 0.808,
) -> pd.DataFrame:
    """
    任意の緯度/経度/方位角/傾斜角で、壁面/ガラス面の日射熱取得量を返す。

    入力（日射）の指定は以下のいずれか:
      - 全天日射量のみ（Erbs 直散分離）
      - 全天日射量 + 法線面直達日射量（Id を復元）
      - 法線面直達日射量 + 水平面拡散日射量（そのまま使用）

    日射モード:
      - "all": 直達 + 拡散 + 反射
      - "diffuse_only": 日陰などを想定し、直達は 0 扱い（反射は拡散由来のみ）
    """
    df_min = _solar_base(
        全天日射量=全天日射量,
        法線面直達日射量=法線面直達日射量,
        水平面拡散日射量=水平面拡散日射量,
        緯度=float(緯度),
        経度=float(経度),
        use_astro=bool(use_astro),
        time_alignment=time_alignment,
        timestamp_ref=timestamp_ref,
        min_sun_alt_deg=float(min_sun_alt_deg),
    )
    out = _solar_gain_by_angles_from_solar_df(
        df_min,
        方位角=方位角,
        傾斜角=傾斜角,
        日射モード=日射モード,
        名前=名前,
        albedo=albedo,
        glass_tau_diffuse=glass_tau_diffuse,
    )
    # デバッグしやすいよう、基礎列も返す
    out["法線面直達日射量 Ib"] = df_min["法線面直達日射量 Ib"]
    out["水平面拡散日射量 Id"] = df_min["水平面拡散日射量 Id"]
    out["太陽高度 hs"] = df_min["太陽高度 hs"]
    out["太陽方位角 AZs"] = df_min["太陽方位角 AZs"]
    return out

__all__ = [
    "sun_loc", "astro_sun_loc",
    "solar_gain_by_angles",
]



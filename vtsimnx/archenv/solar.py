import pandas as pd
import numpy as np

from .solar_position import sun_loc, astro_sun_loc, cos_AZs
from .solar_separation import sep_direct_diffuse
from .solar_shade import _normalize_shade_polygons, _shade_ratio_on_window, _to_window_local_2d


def eta(c):
    """角度補正の近似多項式
    P(c) = 2.3920·c − 3.8636·c^3 + 3.7568·c^5 − 1.3968·c^7
    Horner 形で評価して数値安定性と可読性を両立。
    """
    c2 = c * c
    return c * (2.3920 + c2 * (-3.8636 + c2 * (3.7568 + c2 * (-1.3968))))


def _select_target_surface(out: pd.DataFrame, *, glass: bool) -> pd.DataFrame:
    """壁面/ガラス面の内訳列を統一キーへそろえる。"""
    result = pd.DataFrame(index=out.index)
    result["入射角cos"] = out["入射角cos"]
    if glass:
        result["直達日射量の面成分 Ib"] = out["直達日射量の面成分（ガラス） Ib_g"]
        result["水平面拡散日射量の拡散成分"] = out["水平面拡散日射量の拡散成分（ガラス）"]
        result["水平面拡散日射量の反射成分"] = out["水平面拡散日射量の反射成分（ガラス）"]
        result["日射熱取得量"] = out["日射熱取得量（ガラス）"]
    else:
        result["直達日射量の面成分 Ib"] = out["直達日射量の面成分 Ib"]
        result["水平面拡散日射量の拡散成分"] = out["水平面拡散日射量の拡散成分"]
        result["水平面拡散日射量の反射成分"] = out["水平面拡散日射量の反射成分"]
        result["日射熱取得量"] = out["日射熱取得量（壁面）"]
    return result


def _append_base_columns(result: pd.DataFrame, df_min: pd.DataFrame) -> pd.DataFrame:
    """デバッグ用の基礎列（Ib/Id/hs/AZs）を追記する。"""
    result["法線面直達日射量 Ib"] = df_min["法線面直達日射量 Ib"]
    result["水平面拡散日射量 Id"] = df_min["水平面拡散日射量 Id"]
    result["太陽高度 hs"] = df_min["太陽高度 hs"]
    result["太陽方位角 AZs"] = df_min["太陽方位角 AZs"]
    return result


def _convert_shade_coords_to_window_center(
    shade_coords,
    *,
    window_height: float,
    origin_mode: str,
) -> list[list[tuple[float, float, float]]]:
    """シェード座標を窓中心基準へ正規化する。"""
    shade_polys = _normalize_shade_polygons(shade_coords)
    shade_polys_center: list[list[tuple[float, float, float]]] = []
    for poly in shade_polys:
        poly_center: list[tuple[float, float, float]] = []
        for x, y, z in poly:
            xx, yy = _to_window_local_2d(x, y, window_height=window_height, origin_mode=origin_mode)
            poly_center.append((xx, yy, float(z)))
        shade_polys_center.append(poly_center)
    return shade_polys_center


def _apply_shade_to_direct_components(out: pd.DataFrame, sunlit: pd.Series) -> pd.DataFrame:
    """直達成分へ日向率を掛けて、合計列を再計算する。"""
    out["直達日射量の面成分 Ib"] = out["直達日射量の面成分 Ib"] * sunlit
    out["直達日射量の面成分（ガラス） Ib_g"] = out["直達日射量の面成分（ガラス） Ib_g"] * sunlit
    out["日射熱取得量（壁面）"] = (
        out["直達日射量の面成分 Ib"] + out["水平面拡散日射量の拡散成分"] + out["水平面拡散日射量の反射成分"]
    )
    out["日射熱取得量（ガラス）"] = (
        out["直達日射量の面成分（ガラス） Ib_g"]
        + out["水平面拡散日射量の拡散成分（ガラス）"]
        + out["水平面拡散日射量の反射成分（ガラス）"]
    )
    return out


def _solar_gain_by_angles_from_solar_df(
    df_solar: pd.DataFrame,
    *,
    方位角: float,
    傾斜角: float,
    日射モード: str = "all",
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

    # 拡散/反射（等方モデル + 地面反射の簡易モデル）
    # - Id は水平面拡散（DHI）
    # - Ib は法線面直達（DNI）
    # - 水平面全天（GHI）= DHI + DNI*sin(hs)
    # 傾斜角で view factor:
    #   F_sky   = (1+cosβ)/2, F_ground = (1-cosβ)/2
    f_sky = (1.0 + float(np.cos(beta))) / 2.0
    f_gnd = (1.0 - float(np.cos(beta))) / 2.0
    id_d = id_ * f_sky
    ib_for_ref = ib if 日射モード == "all" else (ib * 0.0)
    ghi = id_ + ib_for_ref * np.maximum(sin_hs, 0.0)
    id_r = ghi * float(albedo) * f_gnd

    wall_gain = ib_surf + id_d + id_r

    # ガラス（直達はeta(cosθ)で角度補正、拡散/反射は一定透過率）
    ib_glass = ib_surf * eta(cos_theta)
    if 日射モード == "diffuse_only":
        ib_glass = ib_glass * 0.0
    id_d_g = id_d * float(glass_tau_diffuse)
    id_r_g = id_r * float(glass_tau_diffuse)
    glass_gain = ib_glass + id_d_g + id_r_g

    out = pd.DataFrame(index=df_solar.index)
    out["入射角cos"] = cos_theta
    out["直達日射量の面成分 Ib"] = ib_surf
    out["水平面拡散日射量の拡散成分"] = id_d
    out["水平面拡散日射量の反射成分"] = id_r
    out["日射熱取得量（壁面）"] = wall_gain

    out["直達日射量の面成分（ガラス） Ib_g"] = ib_glass
    out["水平面拡散日射量の拡散成分（ガラス）"] = id_d_g
    out["水平面拡散日射量の反射成分（ガラス）"] = id_r_g
    out["日射熱取得量（ガラス）"] = glass_gain

    return out


def _solar_base(
    *,
    ghi: pd.Series | None,
    dni: pd.Series | None,
    dhi: pd.Series | None,
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
    if ghi is None and not (dni is not None and dhi is not None):
        raise TypeError("solar_gain_by_angles: ghi または (dni and dhi) を指定してください。")

    idx = (ghi.index if ghi is not None else dni.index)  # type: ignore[union-attr]

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
    if dni is not None and dhi is not None:
        ib = dni.astype("float64")
        id_ = dhi.astype("float64")
        df_min = pd.DataFrame(index=idx)
        df_min["法線面直達日射量 Ib"] = ib
        df_min["水平面拡散日射量 Id"] = id_
        return pd.concat([df_min, df_sun], axis=1)

    if ghi is None:
        raise TypeError("solar_gain_by_angles: ghi がありません。")

    ig = ghi.astype("float64")
    sin_hs = df_sun["太陽高度の正弦 sin_hs"].astype("float64")
    hs = df_sun["太陽高度 hs"].astype("float64")
    day = (hs > float(min_sun_alt_deg)) & (sin_hs > 0)

    # (A) IG + Ib で与えられた場合: Id = IG - Ib*sin(hs) を復元（Ibは上限で丸める）
    if dni is not None:
        ib_in = dni.astype("float64")
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
    ghi: pd.Series | None = None,
    dni: pd.Series | None = None,
    dhi: pd.Series | None = None,
    glass: bool = False,
    return_details: bool = False,
    use_astro: bool = False,
    time_alignment: str = "timestamp",
    timestamp_ref: str = "start",
    min_sun_alt_deg: float = 0.0,
    日射モード: str = "all",
    albedo: float = 0.1,
    glass_tau_diffuse: float = 0.808,
) -> pd.DataFrame | pd.Series:
    """
    任意の緯度/経度/方位角/傾斜角で、壁面またはガラス面の日射熱取得量を返す。

    入力（日射）の指定は以下のいずれか:
      - ghi のみ（Erbs 直散分離）
      - ghi + dni（Id を復元）
      - dni + dhi（そのまま使用）

    glass:
      - False: 壁面の日射熱取得量を返す（既定）
      - True : ガラス面の日射熱取得量を返す

    return_details:
      - False: 日射熱取得量のみ（Series）を返す（既定）
      - True : 詳細列 + 基礎列（Ib/Id/hs/AZs）をDataFrameで返す

    日射モード:
      - "all": 直達 + 拡散 + 反射
      - "diffuse_only": 日陰などを想定し、直達は 0 扱い（反射は拡散由来のみ）
    """
    df_min = _solar_base(
        ghi=ghi,
        dni=dni,
        dhi=dhi,
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
        albedo=albedo,
        glass_tau_diffuse=glass_tau_diffuse,
    )
    result = _select_target_surface(out, glass=glass)

    if not return_details:
        return result["日射熱取得量"].rename("日射熱取得量")

    return _append_base_columns(result, df_min)


def solar_gain_by_angles_with_shade(
    *,
    方位角: float,
    傾斜角: float,
    窓幅: float,
    窓高さ: float,
    シェード座標,
    シェード座標基準: str = "window_center",
    緯度: float = 36.00,
    経度: float = 140.00,
    ghi: pd.Series | None = None,
    dni: pd.Series | None = None,
    dhi: pd.Series | None = None,
    glass: bool = False,
    return_details: bool = False,
    use_astro: bool = False,
    time_alignment: str = "timestamp",
    timestamp_ref: str = "start",
    min_sun_alt_deg: float = 0.0,
    日射モード: str = "all",
    albedo: float = 0.1,
    glass_tau_diffuse: float = 0.808,
) -> pd.DataFrame | pd.Series:
    """
    solar_gain_by_angles に、窓とシェード形状を考慮した直達遮蔽を加えた版。

    - 拡散/反射の計算は solar_gain_by_angles と同一。
    - 直達のみ、窓の被影率 η により (1-η) を掛ける。
    glass:
      - False: 壁面の日射熱取得量を返す（既定）
      - True : ガラス面の日射熱取得量を返す

    return_details:
      - False: 日射熱取得量のみ（Series）を返す（既定）
      - True : 詳細列 + 基礎列（Ib/Id/hs/AZs）をDataFrameで返す


    シェード座標:
      - 単一ポリゴン: list[(x, y, z), ...]（3点以上）
      - 複数ポリゴン: list[list[(x, y, z), ...], ...]
      - 窓ローカル座標 [m]:
          x: 右正, y: 上正, z: 外向き法線方向正（窓面は z=0）
      - シェード座標基準:
          "window_center": 窓中心基準
          "window_top_center": 窓上端中央基準
    """
    if 日射モード not in ("all", "diffuse_only"):
        raise ValueError(f"日射モード must be 'all' or 'diffuse_only', got {日射モード!r}")

    w = float(窓幅)
    h = float(窓高さ)
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"窓幅/窓高さ must be > 0, got {(窓幅, 窓高さ)}")

    df_min = _solar_base(
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        緯度=float(緯度),
        経度=float(経度),
        use_astro=bool(use_astro),
        time_alignment=time_alignment,
        timestamp_ref=timestamp_ref,
        min_sun_alt_deg=float(min_sun_alt_deg),
    )

    shade_polys_center = _convert_shade_coords_to_window_center(
        シェード座標,
        window_height=h,
        origin_mode=シェード座標基準,
    )

    eta_shade = _shade_ratio_on_window(
        azs_deg=df_min["太陽方位角 AZs"].astype("float64"),
        hs_deg=df_min["太陽高度 hs"].astype("float64"),
        surface_az_deg=float(方位角),
        surface_tilt_deg=float(傾斜角),
        window_width=w,
        window_height=h,
        shade_polygons_xyz=shade_polys_center,
    )
    sunlit = 1.0 - eta_shade

    out = _solar_gain_by_angles_from_solar_df(
        df_min,
        方位角=方位角,
        傾斜角=傾斜角,
        日射モード=日射モード,
        albedo=albedo,
        glass_tau_diffuse=glass_tau_diffuse,
    )
    out = _apply_shade_to_direct_components(out, sunlit)
    result = _select_target_surface(out, glass=glass)

    result["被影率η"] = eta_shade
    result["日向率(1-η)"] = sunlit

    if not return_details:
        return result["日射熱取得量"].rename("日射熱取得量")

    return _append_base_columns(result, df_min)

__all__ = [
    "sun_loc", "astro_sun_loc",
    "solar_gain_by_angles",
    "solar_gain_by_angles_with_shade",
]



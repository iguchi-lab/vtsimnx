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


def _to_window_local_2d(
    x: float,
    y: float,
    *,
    window_height: float,
    origin_mode: str,
) -> tuple[float, float]:
    """窓基準の座標を、窓中心基準 (x, y) に変換する。"""
    if origin_mode == "window_center":
        return float(x), float(y)
    if origin_mode == "window_top_center":
        # 上端中央基準: y=0 が窓上端、下向きが負。
        return float(x), float(y) - float(window_height) / 2.0
    raise ValueError(f"shade_origin_mode must be 'window_center' or 'window_top_center', got {origin_mode!r}")


def _poly_area(poly: list[tuple[float, float]]) -> float:
    """2D 多角形の面積（符号なし）"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _clip_poly_halfplane(
    poly: list[tuple[float, float]],
    *,
    inside_fn,
    intersect_fn,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman の 1 half-plane クリップ"""
    if not poly:
        return []
    out: list[tuple[float, float]] = []
    prev = poly[-1]
    prev_in = inside_fn(prev)
    for cur in poly:
        cur_in = inside_fn(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect_fn(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect_fn(prev, cur))
        prev = cur
        prev_in = cur_in
    return out


def _clip_poly_to_rect(
    poly: list[tuple[float, float]],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    """2D 多角形を軸平行矩形にクリップする。"""
    p = poly
    p = _clip_poly_halfplane(
        p,
        inside_fn=lambda q: q[0] >= xmin,
        intersect_fn=lambda a, b: (
            xmin,
            a[1] + (b[1] - a[1]) * (xmin - a[0]) / (b[0] - a[0]),
        ),
    )
    p = _clip_poly_halfplane(
        p,
        inside_fn=lambda q: q[0] <= xmax,
        intersect_fn=lambda a, b: (
            xmax,
            a[1] + (b[1] - a[1]) * (xmax - a[0]) / (b[0] - a[0]),
        ),
    )
    p = _clip_poly_halfplane(
        p,
        inside_fn=lambda q: q[1] >= ymin,
        intersect_fn=lambda a, b: (
            a[0] + (b[0] - a[0]) * (ymin - a[1]) / (b[1] - a[1]),
            ymin,
        ),
    )
    p = _clip_poly_halfplane(
        p,
        inside_fn=lambda q: q[1] <= ymax,
        intersect_fn=lambda a, b: (
            a[0] + (b[0] - a[0]) * (ymax - a[1]) / (b[1] - a[1]),
            ymax,
        ),
    )
    return [(float(x), float(y)) for x, y in p]


def _normalize_shade_polygons(
    shade_coords,
) -> list[list[tuple[float, float, float]]]:
    """
    シェード入力を「複数ポリゴン」形式へ正規化する。
    受理する形:
      - 単一: [(x,y,z), ...]
      - 複数: [[(x,y,z), ...], [(x,y,z), ...], ...]
    """
    if not isinstance(shade_coords, (list, tuple)) or len(shade_coords) == 0:
        raise ValueError("シェード座標 must be non-empty.")

    first = shade_coords[0]
    if (
        isinstance(first, (list, tuple))
        and len(first) == 3
        and not isinstance(first[0], (list, tuple))
    ):
        polys = [shade_coords]
    else:
        polys = shade_coords

    out: list[list[tuple[float, float, float]]] = []
    for poly in polys:
        if not isinstance(poly, (list, tuple)) or len(poly) < 3:
            raise ValueError("各シェードポリゴンは 3 点以上の list/tuple で指定してください。")
        verts: list[tuple[float, float, float]] = []
        for p in poly:
            if not isinstance(p, (list, tuple)) or len(p) != 3:
                raise ValueError("シェード頂点は (x, y, z) の3要素で指定してください。")
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        out.append(verts)
    return out


def _segment_intersection_point(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    eps: float = 1e-12,
) -> tuple[float, float] | None:
    """2線分 AB, CD の交点（単一点）を返す。平行/非交差は None。"""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d

    r = np.array([bx - ax, by - ay], dtype="float64")
    s = np.array([dx - cx, dy - cy], dtype="float64")
    qmp = np.array([cx - ax, cy - ay], dtype="float64")

    den = r[0] * s[1] - r[1] * s[0]
    if abs(float(den)) < eps:
        return None

    t = (qmp[0] * s[1] - qmp[1] * s[0]) / den
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / den
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        x = ax + float(t) * (bx - ax)
        y = ay + float(t) * (by - ay)
        return float(x), float(y)
    return None


def _poly_y_intervals_at_x(
    poly: list[tuple[float, float]],
    x: float,
    eps: float = 1e-12,
) -> list[tuple[float, float]]:
    """多角形と縦線 x=const の交差区間（y区間）を返す。"""
    ys: list[float] = []
    n = len(poly)
    if n < 3:
        return []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx = x2 - x1
        if abs(dx) < eps:
            continue
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        # 頂点二重カウント回避のため半開区間
        if not (xmin <= x < xmax):
            continue
        t = (x - x1) / dx
        y = y1 + t * (y2 - y1)
        ys.append(float(y))

    ys.sort()
    out: list[tuple[float, float]] = []
    m = len(ys)
    for i in range(0, m - 1, 2):
        y0 = ys[i]
        y1 = ys[i + 1]
        if y1 > y0:
            out.append((y0, y1))
    return out


def _union_length_1d(intervals: list[tuple[float, float]], eps: float = 1e-12) -> float:
    """1次元区間の和集合長さ。"""
    if not intervals:
        return 0.0
    segs = sorted(intervals, key=lambda t: t[0])
    cur_l, cur_r = segs[0]
    total = 0.0
    for l, r in segs[1:]:
        if l <= cur_r + eps:
            cur_r = max(cur_r, r)
        else:
            total += max(cur_r - cur_l, 0.0)
            cur_l, cur_r = l, r
    total += max(cur_r - cur_l, 0.0)
    return float(total)


def _union_area_polygons_2d(polys: list[list[tuple[float, float]]], eps: float = 1e-12) -> float:
    """
    複数2Dポリゴンの和集合面積を計算する（重なり二重計上なし）。
    線形境界に対する x-スイープで算出。
    """
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return 0.0

    x_breaks: list[float] = []
    all_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for poly in polys:
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            x_breaks.append(float(a[0]))
            all_edges.append((a, b))

    # エッジ交点の x も分割点に入れて、スラブ内で位相が変わらないようにする
    m = len(all_edges)
    for i in range(m):
        a, b = all_edges[i]
        for j in range(i + 1, m):
            c, d = all_edges[j]
            p = _segment_intersection_point(a, b, c, d, eps=eps)
            if p is not None:
                x_breaks.append(float(p[0]))

    # 近傍値をまとめる
    x_breaks.sort()
    xs: list[float] = []
    for x in x_breaks:
        if not xs or abs(x - xs[-1]) > 1e-10:
            xs.append(x)

    area = 0.0
    for i in range(len(xs) - 1):
        x0 = xs[i]
        x1 = xs[i + 1]
        dx = x1 - x0
        if dx <= eps:
            continue
        xm = (x0 + x1) * 0.5
        intervals: list[tuple[float, float]] = []
        for poly in polys:
            intervals.extend(_poly_y_intervals_at_x(poly, xm, eps=eps))
        area += _union_length_1d(intervals, eps=eps) * dx
    return float(max(area, 0.0))


def _shade_ratio_on_window(
    *,
    azs_deg: pd.Series,
    hs_deg: pd.Series,
    surface_az_deg: float,
    surface_tilt_deg: float,
    window_width: float,
    window_height: float,
    shade_polygons_xyz: list[list[tuple[float, float, float]]],
) -> pd.Series:
    """
    各時刻における窓面の被影率 η（0..1）を返す。
    shade_polygons_xyz は窓ローカル座標:
      x: 右正, y: 上正, z: 外向き法線方向正（窓面は z=0）
    """
    if window_width <= 0.0 or window_height <= 0.0:
        raise ValueError("window_width/window_height must be > 0.")
    if len(shade_polygons_xyz) == 0:
        raise ValueError("shade_polygons_xyz must have at least 1 polygon.")

    gamma = np.radians(float(surface_az_deg))
    beta = np.radians(float(surface_tilt_deg))

    n = np.array(
        [
            -np.sin(beta) * np.sin(gamma),
            -np.sin(beta) * np.cos(gamma),
            np.cos(beta),
        ],
        dtype="float64",
    )

    up = np.array([0.0, 0.0, 1.0], dtype="float64")
    u = np.cross(up, n)
    if np.linalg.norm(u) < 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype="float64")
    else:
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    az = np.radians(azs_deg.to_numpy(dtype="float64"))
    hs = np.radians(hs_deg.to_numpy(dtype="float64"))
    s_e = -np.cos(hs) * np.sin(az)
    s_n = -np.cos(hs) * np.cos(az)
    s_u = np.sin(hs)
    s_global = np.stack([s_e, s_n, s_u], axis=1)

    su = s_global @ u
    sv = s_global @ v
    sn = s_global @ n

    xmin = -float(window_width) / 2.0
    xmax = float(window_width) / 2.0
    ymin = -float(window_height) / 2.0
    ymax = float(window_height) / 2.0
    area_window = float(window_width) * float(window_height)

    eta_arr = np.zeros(len(azs_deg), dtype="float64")
    for i in range(len(eta_arr)):
        if sn[i] <= 1e-9:
            eta_arr[i] = 0.0
            continue

        clipped_polys: list[list[tuple[float, float]]] = []
        for poly_xyz in shade_polygons_xyz:
            proj_poly: list[tuple[float, float]] = []
            for px, py, pz in poly_xyz:
                x = float(px) - float(pz) * float(su[i]) / float(sn[i])
                y = float(py) - float(pz) * float(sv[i]) / float(sn[i])
                proj_poly.append((x, y))

            clipped = _clip_poly_to_rect(proj_poly, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            if len(clipped) >= 3 and _poly_area(clipped) > 0.0:
                clipped_polys.append(clipped)
        a_union = _union_area_polygons_2d(clipped_polys)
        eta_arr[i] = min(max(a_union / area_window, 0.0), 1.0)

    return pd.Series(eta_arr, index=azs_deg.index, dtype="float64")


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
    time_alignment: str = "timestamp",
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
    全天日射量: pd.Series | None = None,
    法線面直達日射量: pd.Series | None = None,
    水平面拡散日射量: pd.Series | None = None,
    名前: str = "任意面",
    use_astro: bool = False,
    time_alignment: str = "timestamp",
    timestamp_ref: str = "start",
    min_sun_alt_deg: float = 0.0,
    日射モード: str = "all",
    albedo: float = 0.1,
    glass_tau_diffuse: float = 0.808,
) -> pd.DataFrame:
    """
    solar_gain_by_angles に、窓とシェード形状を考慮した直達遮蔽を加えた版。

    - 拡散/反射の計算は solar_gain_by_angles と同一。
    - 直達のみ、窓の被影率 η により (1-η) を掛ける。

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

    shade_polys = _normalize_shade_polygons(シェード座標)
    shade_polys_center: list[list[tuple[float, float, float]]] = []
    for poly in shade_polys:
        poly_center: list[tuple[float, float, float]] = []
        for x, y, z in poly:
            xx, yy = _to_window_local_2d(x, y, window_height=h, origin_mode=シェード座標基準)
            poly_center.append((xx, yy, float(z)))
        shade_polys_center.append(poly_center)

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
        名前=名前,
        albedo=albedo,
        glass_tau_diffuse=glass_tau_diffuse,
    )

    col_ib = f"直達日射量の{名前}面成分 Ib"
    col_ibg = f"直達日射量の{名前}面成分（ガラス） Ib_g"
    col_idd = f"水平面拡散日射量の拡散成分（{名前}）"
    col_idr = f"水平面拡散日射量の反射成分（{名前}）"
    col_iddg = f"水平面拡散日射量の拡散成分（ガラス）（{名前}）"
    col_idrg = f"水平面拡散日射量の反射成分（ガラス）（{名前}）"
    col_wall = f"日射熱取得量（{名前}）"
    col_glass = f"日射熱取得量（{名前}ガラス）"

    out[col_ib] = out[col_ib] * sunlit
    out[col_ibg] = out[col_ibg] * sunlit
    out[col_wall] = out[col_ib] + out[col_idd] + out[col_idr]
    out[col_glass] = out[col_ibg] + out[col_iddg] + out[col_idrg]

    out[f"被影率η（{名前}）"] = eta_shade
    out[f"日向率(1-η)（{名前}）"] = sunlit

    out["法線面直達日射量 Ib"] = df_min["法線面直達日射量 Ib"]
    out["水平面拡散日射量 Id"] = df_min["水平面拡散日射量 Id"]
    out["太陽高度 hs"] = df_min["太陽高度 hs"]
    out["太陽方位角 AZs"] = df_min["太陽方位角 AZs"]
    return out

__all__ = [
    "sun_loc", "astro_sun_loc",
    "solar_gain_by_angles",
    "solar_gain_by_angles_with_shade",
]



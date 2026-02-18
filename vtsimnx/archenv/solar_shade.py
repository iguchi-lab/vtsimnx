from __future__ import annotations

import numpy as np
import pandas as pd


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


__all__ = [
    "_to_window_local_2d",
    "_normalize_shade_polygons",
    "_shade_ratio_on_window",
]


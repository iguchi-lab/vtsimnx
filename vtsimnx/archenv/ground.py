from __future__ import annotations

import numpy as np
import pandas as pd


def _as_depth_array(depth_m: float | list[float] | np.ndarray) -> np.ndarray:
    if np.isscalar(depth_m):
        arr = np.array([float(depth_m)], dtype="float64")
    else:
        arr = np.asarray(depth_m, dtype="float64")
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("depth_m must be a scalar or 1D non-empty array-like.")
    if np.any(arr < 0.0):
        raise ValueError("depth_m must be >= 0.")
    return arr


def _ensure_aligned_index(
    t_out: pd.Series,
    solar_horizontal: pd.Series | None,
    nocturnal_horizontal: pd.Series | None,
) -> pd.DatetimeIndex:
    if not isinstance(t_out.index, pd.DatetimeIndex):
        raise TypeError("t_out index must be DatetimeIndex.")
    idx = t_out.index
    if solar_horizontal is not None and not solar_horizontal.index.equals(idx):
        raise ValueError("solar_horizontal index must match t_out index.")
    if nocturnal_horizontal is not None and not nocturnal_horizontal.index.equals(idx):
        raise ValueError("nocturnal_horizontal index must match t_out index.")
    if len(idx) < 2:
        raise ValueError("At least 2 timesteps are required.")
    return idx


def _dt_seconds(idx: pd.DatetimeIndex) -> float:
    dt = np.diff(idx.view("int64")) / 1e9
    if not np.allclose(dt, dt[0]):
        raise ValueError("Index interval must be constant.")
    if dt[0] <= 0:
        raise ValueError("Index must be strictly increasing.")
    return float(dt[0])


def _prepare_thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = diag.size
    c_prime = np.zeros(max(n - 1, 0), dtype="float64")
    denom = np.zeros(n, dtype="float64")
    denom[0] = diag[0]
    for i in range(n - 1):
        c_prime[i] = upper[i] / denom[i]
        denom[i + 1] = diag[i + 1] - lower[i] * c_prime[i]
    return c_prime, denom


def _solve_thomas_prepared(
    lower: np.ndarray,
    c_prime: np.ndarray,
    denom: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = rhs.size
    d_prime = np.zeros(n, dtype="float64")
    d_prime[0] = rhs[0] / denom[0]
    for i in range(1, n):
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom[i]
    x = np.zeros(n, dtype="float64")
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def ground_temperature_by_depth(
    *,
    depth_m: float | list[float] | np.ndarray,
    t_out: pd.Series,
    solar_horizontal: pd.Series | None = None,
    nocturnal_horizontal: pd.Series | None = None,
    deep_layer_depth_m: float = 10.0,
    deep_layer_temp_c: float = 10.0,
    thermal_conductivity_w_mk: float = 1.5,
    volumetric_heat_capacity_j_m3k: float = 2.2e6,
    solar_to_surface_temp_coeff: float = 0.0,
    nocturnal_to_surface_temp_coeff: float = 0.0,
    model_depth_m: float | None = None,
    n_grid: int = 121,
    init_temp_c: float | None = None,
    spinup: bool = False,
    spinup_cycles: int = 5,
    return_details: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    気象時系列から、任意深さの地盤温度を 1 次元熱伝導で推定する。

    モデル:
      - z=0（地表）: 等価地表温度 Ts(t) を Dirichlet 境界として与える
      - z=H（深部）: 不易層温度 deep_layer_temp_c を固定
      - 地中: ∂T/∂t = alpha * ∂²T/∂z²
      - alpha = k / Cvol

    等価地表温度:
      Ts = t_out
           + solar_to_surface_temp_coeff * solar_horizontal
           - nocturnal_to_surface_temp_coeff * nocturnal_horizontal

    Notes:
      - solar_horizontal / nocturnal_horizontal が None の場合、その項は 0 扱い。
      - 熱容量は体積熱容量 [J/m3/K] を指定する。
      - depth_m がスカラーのとき既定戻り値は Series、複数深さは DataFrame。
      - return_details=True で `地表等価温度` 列を含む DataFrame を返す。
      - spinup=True のとき、同じ入力気象を `spinup_cycles` 回繰り返して計算し、
        最終周期（1 周期分）のみを返す。
    """
    depths = _as_depth_array(depth_m)
    idx = _ensure_aligned_index(t_out, solar_horizontal, nocturnal_horizontal)
    dt = _dt_seconds(idx)

    if deep_layer_depth_m <= 0.0:
        raise ValueError("deep_layer_depth_m must be > 0.")
    if thermal_conductivity_w_mk <= 0.0:
        raise ValueError("thermal_conductivity_w_mk must be > 0.")
    if volumetric_heat_capacity_j_m3k <= 0.0:
        raise ValueError("volumetric_heat_capacity_j_m3k must be > 0.")
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3.")
    if spinup and spinup_cycles < 2:
        raise ValueError("spinup_cycles must be >= 2 when spinup=True.")

    z_max = float(np.max(depths))
    domain_depth = float(model_depth_m) if model_depth_m is not None else max(z_max, float(deep_layer_depth_m))
    if domain_depth <= 0.0:
        raise ValueError("model_depth_m must be > 0.")
    if domain_depth < z_max:
        raise ValueError("model_depth_m must be >= max(depth_m).")

    solar = 0.0 if solar_horizontal is None else solar_horizontal.astype("float64")
    noct = 0.0 if nocturnal_horizontal is None else nocturnal_horizontal.astype("float64")
    ts = (
        t_out.astype("float64")
        + float(solar_to_surface_temp_coeff) * solar
        - float(nocturnal_to_surface_temp_coeff) * noct
    )

    z = np.linspace(0.0, domain_depth, int(n_grid), dtype="float64")
    dz = float(z[1] - z[0])
    alpha = float(thermal_conductivity_w_mk) / float(volumetric_heat_capacity_j_m3k)
    fo = alpha * dt / (dz * dz)

    n_inner = int(n_grid) - 2
    lower = np.full(n_inner - 1, -fo, dtype="float64")
    diag = np.full(n_inner, 1.0 + 2.0 * fo, dtype="float64")
    upper = np.full(n_inner - 1, -fo, dtype="float64")
    c_prime, denom = _prepare_thomas(lower, diag, upper)

    ts_np_cycle = ts.to_numpy(dtype="float64")
    if spinup:
        ts_np = np.tile(ts_np_cycle, int(spinup_cycles))
    else:
        ts_np = ts_np_cycle

    n_total = int(ts_np.size)
    n_out = len(idx)
    out_start = n_total - n_out

    t_arr = np.empty((n_total, int(n_grid)), dtype="float64")
    t0 = float(ts_np[0] if init_temp_c is None else init_temp_c)
    t_arr[0, :] = np.linspace(t0, float(deep_layer_temp_c), int(n_grid))
    t_arr[0, 0] = float(ts_np[0])
    t_arr[0, -1] = float(deep_layer_temp_c)

    for n in range(1, n_total):
        prev = t_arr[n - 1]
        rhs = prev[1:-1].copy()
        rhs[0] += fo * ts_np[n]
        rhs[-1] += fo * float(deep_layer_temp_c)
        inner = _solve_thomas_prepared(lower, c_prime, denom, rhs)
        t_arr[n] = prev
        t_arr[n, 0] = ts_np[n]
        t_arr[n, -1] = float(deep_layer_temp_c)
        t_arr[n, 1:-1] = inner

    target = np.empty((n_out, len(depths)), dtype="float64")
    for i in range(n_out):
        target[i, :] = np.interp(depths, z, t_arr[out_start + i, :])
    ts_out = pd.Series(ts_np[out_start:], index=idx, dtype="float64")

    if len(depths) == 1:
        s = pd.Series(target[:, 0], index=idx, name="地盤温度")
        if not return_details:
            return s
        out = pd.DataFrame(index=idx)
        out["地表等価温度"] = ts_out
        out["地盤温度"] = s
        return out

    out = pd.DataFrame(index=idx)
    for j, d in enumerate(depths):
        out[f"地盤温度_{d:.3f}m"] = target[:, j]
    if not return_details:
        return out
    out.insert(0, "地表等価温度", ts_out)
    return out


__all__ = ["ground_temperature_by_depth"]


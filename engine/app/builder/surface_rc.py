from __future__ import annotations

import numpy as np


def _layers_to_rc_arrays(layers: list[dict]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    層リストから物性配列と抵抗・容量を算出する。
    返り値: (n, lam, thk, vc, C, R_half, R_between)。
    """
    if not layers:
        raise ValueError("layers is empty")
    n = len(layers)
    lam = np.array([float(x["lambda"]) for x in layers], dtype=float)
    thk = np.array([float(x["t"]) for x in layers], dtype=float)
    vc = np.array([float(x["v_capa"]) for x in layers], dtype=float)
    if np.any(lam <= 0) or np.any(thk <= 0) or np.any(vc < 0):
        raise ValueError("invalid layer properties: lambda>0, t>0, v_capa>=0 required")
    C = vc * thk
    R_half = (thk / 2.0) / lam
    R_between = np.zeros(max(n - 1, 0), dtype=float)
    for i in range(n - 1):
        R_between[i] = R_half[i] + R_half[i + 1]
    return n, lam, thk, vc, C, R_half, R_between


def _build_rc_continuous_abcd(
    n: int, C: np.ndarray, R_half: np.ndarray, R_between: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RC 連鎖の連続系状態空間 (A, B, Cmat, Dmat) を構築する。
    u=[Ts, Tt], y=[q''src, q''tgt]。
    """
    def invR(r: float) -> float:
        return 0.0 if r == 0 else 1.0 / r

    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, 2), dtype=float)
    g_s = invR(R_half[0])
    if n == 1:
        g_t = invR(R_half[0])
        A[0, 0] = -(g_s + g_t) / C[0] if C[0] > 0 else 0.0
        B[0, 0] = g_s / C[0] if C[0] > 0 else 0.0
        B[0, 1] = g_t / C[0] if C[0] > 0 else 0.0
    else:
        g_12 = invR(R_between[0])
        A[0, 0] = -(g_s + g_12) / C[0] if C[0] > 0 else 0.0
        A[0, 1] = g_12 / C[0] if C[0] > 0 else 0.0
        B[0, 0] = g_s / C[0] if C[0] > 0 else 0.0
        for i in range(1, n - 1):
            g_im1 = invR(R_between[i - 1])
            g_ip1 = invR(R_between[i])
            if C[i] > 0:
                A[i, i - 1] = g_im1 / C[i]
                A[i, i] = -(g_im1 + g_ip1) / C[i]
                A[i, i + 1] = g_ip1 / C[i]
        g_t = invR(R_half[-1])
        g_nm1 = invR(R_between[-1])
        if C[-1] > 0:
            A[-1, -2] = g_nm1 / C[-1]
            A[-1, -1] = -(g_nm1 + g_t) / C[-1]
            B[-1, 1] = g_t / C[-1]

    Cmat = np.zeros((2, n), dtype=float)
    Dmat = np.zeros((2, 2), dtype=float)
    Cmat[0, 0] = -invR(R_half[0])
    Dmat[0, 0] = invR(R_half[0])
    Cmat[1, -1] = -invR(R_half[-1])
    Dmat[1, 1] = invR(R_half[-1])
    return A, B, Cmat, Dmat

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import pandas as pd

from sitecal.tbc_default_tm import TBCDefaultTM


@dataclass(frozen=True)
class HorizontalCal:
    """Simple horizontal mapping: LOCAL_EN = TM_EN + (dE, dN)."""
    dE: float
    dN: float
    rmse_m: float


@dataclass(frozen=True)
class Similarity2D:
    a: float
    b: float
    tE: float
    tN: float
    rmse_m: float

    def apply(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> tuple:
        """Vectorized application of the similarity transformation."""
        E = self.a * x - self.b * y + self.tE
        N = self.b * x + self.a * y + self.tN
        return E, N

    def inverse_apply(self, E: Union[float, np.ndarray], N: Union[float, np.ndarray]) -> tuple:
        """Vectorized inverse application: Local (E,N) -> TM (x,y)."""
        dx = E - self.tE
        dy = N - self.tN
        denom = (self.a * self.a) + (self.b * self.b)
        x = (self.a * dx + self.b * dy) / denom
        y = (-self.b * dx + self.a * dy) / denom
        return x, y


def solve_horizontal_shift(points: Iterable, tm: TBCDefaultTM) -> HorizontalCal:
    pts = list(points)
    if not pts:
        raise ValueError("No points")

    # Vectorized extraction
    lons = np.array([p.lon_deg for p in pts])
    lats = np.array([p.lat_deg for p in pts])
    loc_e = np.array([p.local_e_m for p in pts])
    loc_n = np.array([p.local_n_m for p in pts])

    # Vectorized transform (pyproj supports arrays)
    tm_e, tm_n = tm.to_tm.transform(lons, lats)

    # Least squares for constant shift:
    dE = np.mean(loc_e - tm_e)
    dN = np.mean(loc_n - tm_n)

    # RMSE in metres:
    de_res = (tm_e + dE) - loc_e
    dn_res = (tm_n + dN) - loc_n
    rmse = np.sqrt(np.mean(de_res**2 + dn_res**2))

    return HorizontalCal(dE=float(dE), dN=float(dN), rmse_m=float(rmse))


def solve_similarity_2d(points: Iterable, tm: TBCDefaultTM) -> Similarity2D:
    pts = list(points)
    if not pts:
        raise ValueError("No points")

    lons = np.array([p.lon_deg for p in pts])
    lats = np.array([p.lat_deg for p in pts])
    loc_e = np.array([p.local_e_m for p in pts])
    loc_n = np.array([p.local_n_m for p in pts])

    tm_e, tm_n = tm.to_tm.transform(lons, lats)
    n = len(pts)

    # Build system A * x = L using slicing for speed
    A = np.zeros((2 * n, 4))
    L = np.zeros(2 * n)

    # E equations: [x, -y, 1, 0]
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = tm_e, -tm_n, 1.0
    L[0::2] = loc_e

    # N equations: [y, x, 0, 1]
    A[1::2, 0], A[1::2, 1], A[1::2, 3] = tm_n, tm_e, 1.0
    L[1::2] = loc_n

    # Least squares
    x_hat, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
    a, b, tE, tN = x_hat

    residuals = A @ x_hat - L
    rmse = np.sqrt(np.mean(residuals**2))

    return Similarity2D(a=float(a), b=float(b), tE=float(tE), tN=float(tN), rmse_m=float(rmse))


def apply_local_to_global_vectorized(df: pd.DataFrame, sim: Similarity2D, tm: TBCDefaultTM) -> pd.DataFrame:
    """
    Optimized transformation using NumPy arrays. No for loops.
    """
    # 1. Local EN -> TM EN (Inverse Similarity)
    tm_e, tm_n = sim.inverse_apply(df['e'].values, df['n'].values)
    # 2. TM EN -> WGS84 (Inverse Projection)
    df['lon'], df['lat'] = tm.to_geo.transform(tm_e, tm_n)
    return df
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from sitecal.tbc_default_tm import TBCDefaultTM


@dataclass(frozen=True)
class HorizontalCal:
    """Simple horizontal mapping: LOCAL_EN = TM_EN + (dE, dN)."""
    dE: float
    dN: float
    rmse_m: float


def solve_horizontal_shift(points: Iterable, tm: TBCDefaultTM) -> HorizontalCal:
    pts = list(points)
    if not pts:
        raise ValueError("No points")

    tm_e = []
    tm_n = []
    loc_e = []
    loc_n = []

    for p in pts:
        x, y = tm.to_tm.transform(p.lon_deg, p.lat_deg)
        tm_e.append(x)
        tm_n.append(y)
        loc_e.append(p.local_e_m)
        loc_n.append(p.local_n_m)

    tm_e = np.array(tm_e, dtype=float)
    tm_n = np.array(tm_n, dtype=float)
    loc_e = np.array(loc_e, dtype=float)
    loc_n = np.array(loc_n, dtype=float)

    # Least squares for constant shift:
    dE = float(np.mean(loc_e - tm_e))
    dN = float(np.mean(loc_n - tm_n))

    # RMSE in metres:
    de_res = (tm_e + dE) - loc_e
    dn_res = (tm_n + dN) - loc_n
    rmse = float(np.sqrt(np.mean(de_res**2 + dn_res**2)))

    return HorizontalCal(dE=dE, dN=dN, rmse_m=rmse)

@dataclass(frozen=True)
class Similarity2D:
    a: float
    b: float
    tE: float
    tN: float
    rmse_m: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        E = self.a * x - self.b * y + self.tE
        N = self.b * x + self.a * y + self.tN
        return E, N
    
    def inverse_apply(self, E: float, N: float) -> tuple[float, float]:
        """
        Invert:
        [E] = [a -b][x] + [tE]
        [N]   [b  a][y]   [tN]
        Return (x, y) in TM metres.
        """
        dE = E - self.tE
        dN = N - self.tN
        det = self.a * self.a + self.b * self.b

        x = ( self.a * dE + self.b * dN) / det
        y = (-self.b * dE + self.a * dN) / det
        return x, y
    
    def inverse_apply(self, E: float, N: float) -> tuple[float, float]:
        """
        Local (E,N) -> shifted TM (x,y), inverse of:
        E = tE + a*x - b*y
        N = tN + b*x + a*y
        """
        dx = E - self.tE
        dy = N - self.tN
        denom = (self.a * self.a) + (self.b * self.b)
        x = ( self.a * dx + self.b * dy) / denom
        y = (-self.b * dx + self.a * dy) / denom
        return x, y


def solve_similarity_2d(points: Iterable, tm: TBCDefaultTM) -> Similarity2D:
    pts = list(points)
    if not pts:
        raise ValueError("No points")

    A_rows = []
    L = []

    for p in pts:
        x, y = tm.to_tm.transform(p.lon_deg, p.lat_deg)
        E, N = p.local_e_m, p.local_n_m

        # E eq:  [x, -y, 1, 0] [a, b, tE, tN]^T = E
        A_rows.append([x, -y, 1.0, 0.0])
        L.append(E)

        # N eq:  [y,  x, 0, 1] [a, b, tE, tN]^T = N
        A_rows.append([y,  x, 0.0, 1.0])
        L.append(N)

    A = np.array(A_rows, dtype=float)
    L = np.array(L, dtype=float)

    # Least squares
    x_hat, *_ = np.linalg.lstsq(A, L, rcond=None)
    a, b, tE, tN = map(float, x_hat)

    # RMSE
    residuals = A @ x_hat - L
    residuals = residuals.reshape(-1, 1)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return Similarity2D(a=a, b=b, tE=tE, tN=tN, rmse_m=rmse)
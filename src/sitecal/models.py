from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CRSConfig:
    """
    Coordinate system config.

    mode:
      - "tbc_default": Trimble "Default projection (Transverse Mercator)" behaviour:
          * TM centred at first point, k0=1, x0=y0=0
          * all TM coords shifted so first point becomes (0,0)
          * horizontal origin E0/N0 is centroid of shifted TM coords
      - "utm": explicit UTM zone (fallback)
    """
    mode: str = "tbc_default"
    utm_zone: Optional[int] = None
    hemisphere: str = "S"  # "N" or "S"


@dataclass(frozen=True)
class VerticalModel:
    mode: str = "none"  # "none" | "shift" | "tilt"


@dataclass(frozen=True)
class CalibrationConfig:
    solve_scale: bool = True
    vertical: VerticalModel = field(default_factory=VerticalModel)

    # Vertical plane origin in LOCAL coords (optional).
    # If None, we anchor at first usable point (TBC-ish behaviour in our pipeline).
    vertical_origin_E: Optional[float] = None
    vertical_origin_N: Optional[float] = None

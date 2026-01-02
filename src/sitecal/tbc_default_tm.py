from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class TBCDefaultTM:
    """
    Trimble-ish 'Default projection (Transverse Mercator)'.
    We approximate it as:
      - TM centred at first point (lon0/lat0)
      - k0=1, x0=y0=0
    Returns a pyproj Transformer for (lon,lat) <-> (E,N).
    """
    lat0: float
    lon0: float
    crs: CRS
    to_tm: Transformer
    to_geo: Transformer


def build_tbc_default_tm(points: Iterable) -> TBCDefaultTM:
    pts = list(points)
    if not pts:
        raise ValueError("No points provided")

    p0 = pts[0]
    lat0 = float(p0.lat_deg)
    lon0 = float(p0.lon_deg)

    # TM centred at first point. k=1 to mimic Trimble default.
    crs = CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    )

    # Always use lon/lat ordering explicitly.
    to_tm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    return TBCDefaultTM(lat0=lat0, lon0=lon0, crs=crs, to_tm=to_tm, to_geo=to_geo)
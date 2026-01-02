from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from bs4 import BeautifulSoup
from sitecal.angles import dms_to_decimal


@dataclass(frozen=True)
class ControlPoint:
    """Minimal control point mapping we need for calibration."""
    point_id: str
    lat_deg: float
    lon_deg: float
    h_ell_m: float
    local_e_m: float
    local_n_m: float
    local_h_m: float


def _parse_m_value(s: str) -> float:
    # "18276.544 m" -> 18276.544
    return float(s.replace("m", "").strip())


def _table_to_kv(table) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    TBC point tables look like:
      ['Point', 'p55']
      ['Latitude', 'S24°...']
      ['Longitude', 'W69°...']
      ['Height', '3179.592 m']
    or:
      ['Point', 'p55-L']
      ['Easting', '19842.893 m']
      ['Northing', '114398.129 m']
      ['Elevation', '3143.547 m']
    """
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)

    # We only care about 2-col key/value tables
    if len(rows) < 2:
        return None
    if any(len(r) != 2 for r in rows[:4]):
        return None
    if rows[0][0] != "Point":
        return None

    point = rows[0][1]
    kv = {k: v for (k, v) in rows[1:] if k and v}
    return point, kv


def extract_control_points_from_tbc_report(path: str | Path) -> List[ControlPoint]:
    """
    Extract control points by pairing:
      - base point geodetic (Point=pXX with Latitude/Longitude/Height)
      - local point ENH (Point=pXX-L with Easting/Northing/Elevation)
    """
    p = Path(path)
    soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "lxml")

    geo: Dict[str, Dict[str, str]] = {}
    enh: Dict[str, Dict[str, str]] = {}

    for table in soup.find_all("table"):
        parsed = _table_to_kv(table)
        if not parsed:
            continue
        point, kv = parsed

        keys = set(kv.keys())
        if {"Latitude", "Longitude", "Height"}.issubset(keys):
            geo[point] = kv
        if {"Easting", "Northing"}.issubset(keys) and ("Elevation" in keys or "Height" in keys):
            enh[point] = kv

    out: List[ControlPoint] = []
    for base, g in geo.items():
        local_name = f"{base}-L"
        if local_name not in enh:
            continue

        l = enh[local_name]
        h_key = "Elevation" if "Elevation" in l else "Height"
        lat_dms = g["Latitude"]
        lon_dms = g["Longitude"]

        out.append(
            ControlPoint(
                point_id=base,
                lat_deg=dms_to_decimal(lat_dms),
                lon_deg=dms_to_decimal(lon_dms),
                h_ell_m=_parse_m_value(g["Height"]),
                local_e_m=_parse_m_value(l["Easting"]),
                local_n_m=_parse_m_value(l["Northing"]),
                local_h_m=_parse_m_value(l[h_key]),
            )
        )
        

    if not out:
        raise ValueError("No control points found. TBC report format may differ from expected key/value tables.")
    return out
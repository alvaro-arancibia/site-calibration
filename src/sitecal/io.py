# src/sitecal/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import csv
import pandas as pd


@dataclass(frozen=True)
class ControlPoint:
    id: str

    # LOCAL (site) coordinates
    E: float
    N: float
    M: Optional[float] = None  # local elevation (optional)

    # GNSS / global geodetic (from TBC report)
    lon: Optional[float] = None
    lat: Optional[float] = None
    h: Optional[float] = None  # ellipsoidal height
    H: Optional[float] = None  # orthometric/local height if present

    # weights (TBC-ish defaults)
    w_h: float = 1.0
    w_v: float = 1.0


def read_local_csv(path: str | Path) -> list[ControlPoint]:
    """
    Reads a CSV with at least: id,E,N
    Optionally: M (or m)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    points: list[ControlPoint] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        # normalise header keys
        def get(row, *keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return None

        for row in reader:
            pid = get(row, "id", "ID", "Id", "name", "Name")
            if not pid:
                raise ValueError("Missing 'id' column value in a row.")

            E = float(get(row, "E", "e"))
            N = float(get(row, "N", "n"))

            m = get(row, "M", "m")
            M = float(m) if m is not None else None

            points.append(ControlPoint(id=str(pid), E=E, N=N, M=M))

    return points


def read_csv_to_dataframe(path: str | Path) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame and normalizes column names.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    
    # Normalize column names to a consistent format
    column_mapping = {
        "id": "Point", "ID": "Point", "Id": "Point", "name": "Point", "Name": "Point",
        "lon": "Lon", "longitude": "Lon",
        "lat": "Lat", "latitude": "Lat",
        "h": "h", "ellipsoidal_height": "h",
        "E": "Easting", "easting": "Easting",
        "N": "Northing", "northing": "Northing",
        "M": "h_local", "m": "h_local", "elevation": "h_local", "H": "h_local"
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Ensure Point column is string
    if "Point" in df.columns:
        df["Point"] = df["Point"].astype(str)
        
    return df
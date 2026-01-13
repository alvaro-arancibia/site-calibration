from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from sitecal.io import ControlPoint


def read_local_csv(path: str | Path) -> List[ControlPoint]:
    """Reads a CSV with local coordinates using Pandas for robust parsing."""
    df = pd.read_csv(path)
    # Normalize column names to lowercase to be case-insensitive
    df.columns = [c.lower() for c in df.columns]

    points: List[ControlPoint] = []
    for _, row in df.iterrows():
        # Match ID or Name
        pid = row.get("id") or row.get("name")
        if pid is None:
            continue

        # Match E, N, M (Elevation)
        e = row.get("e")
        n = row.get("n")
        m = row.get("m")

        points.append(
            ControlPoint(
                id=str(pid),
                E=float(e),
                N=float(n),
                M=float(m) if pd.notnull(m) else None,
            )
        )
    return points


def save_results_csv(path: str | Path, df: pd.DataFrame) -> None:
    """Saves a Pandas DataFrame to CSV."""
    df.to_csv(path, index=False)
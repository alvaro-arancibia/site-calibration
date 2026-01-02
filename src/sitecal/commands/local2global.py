from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional, Tuple

import numpy as np
import typer

from sitecal.tbc_html import extract_control_points_from_tbc_report
from sitecal.tbc_default_tm import build_tbc_default_tm
from sitecal.calibration import solve_similarity_2d


@dataclass(frozen=True)
class Row:
    id: str
    E: float
    N: float
    H: float  # local height (your CSV might call this H or M)


def _read_local_csv(path: Path) -> list[Row]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        headers = set(rdr.fieldnames or [])

        if "id" not in headers or "E" not in headers or "N" not in headers:
            raise ValueError(f"CSV must include headers: id,E,N (got {rdr.fieldnames})")

        # height column can be M or H
        h_col = "M" if "M" in headers else ("H" if "H" in headers else None)
        if h_col is None:
            raise ValueError(f"CSV must include a height column named M or H (got {rdr.fieldnames})")

        out: list[Row] = []
        for r in rdr:
            out.append(
                Row(
                    id=str(r["id"]).strip(),
                    E=float(r["E"]),
                    N=float(r["N"]),
                    H=float(r[h_col]),
                )
            )
        return out


def _write_out_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fit_vertical_plane(ctrl) -> Tuple[float, float, float]:
    """
    Fit dh(E,N) = c0 + cE*E + cN*N  where:
      h_ellip ≈ local_h + dh(E,N)
    """
    E = np.array([p.local_e_m for p in ctrl], dtype=float)
    N = np.array([p.local_n_m for p in ctrl], dtype=float)
    dh = np.array([p.h_ell_m - p.local_h_m for p in ctrl], dtype=float)

    A = np.column_stack([np.ones_like(E), E, N])
    coef, *_ = np.linalg.lstsq(A, dh, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def run(
    cal_report: Path,
    input_csv: Path,
    output_csv: Path,
    paranoia: bool = True,
) -> None:
    ctrl = extract_control_points_from_tbc_report(str(cal_report))
    if len(ctrl) < 3:
        raise ValueError(f"Need >=3 control points, got {len(ctrl)}")

    tm = build_tbc_default_tm(ctrl)
    sim = solve_similarity_2d(ctrl, tm)

    c0, cE, cN = _fit_vertical_plane(ctrl)

    def dh_at(E: float, N: float) -> float:
        return c0 + cE * E + cN * N

    rows = _read_local_csv(input_csv)
    out_rows: list[dict] = []

    max_dE = 0.0
    max_dN = 0.0
    max_dH = 0.0
    worst: Optional[tuple] = None

    for r in rows:
        x_tm, y_tm = sim.inverse_apply(E=r.E, N=r.N)
        lon, lat = tm.to_geo.transform(x_tm, y_tm)
        h_ellip = r.H + dh_at(r.E, r.N)

        out_rows.append(
            {
                "id": r.id,
                "E": f"{r.E:.3f}",
                "N": f"{r.N:.3f}",
                "H": f"{r.H:.3f}",
                "lon": f"{lon:.10f}",
                "lat": f"{lat:.10f}",
                "h": f"{h_ellip:.4f}",
            }
        )

        if paranoia:
            E2, N2 = sim.apply(x_tm, y_tm)
            H2 = h_ellip - dh_at(r.E, r.N)

            dE = abs(E2 - r.E) * 1000.0
            dN = abs(N2 - r.N) * 1000.0
            dH = abs(H2 - r.H) * 1000.0

            if (dE > max_dE) or (dN > max_dN) or (dH > max_dH):
                worst = (r.id, dE, dN, dH)

            max_dE = max(max_dE, dE)
            max_dN = max(max_dN, dN)
            max_dH = max(max_dH, dH)

    _write_out_csv(output_csv, out_rows)
    typer.echo(f"Wrote {len(out_rows)} points to {output_csv}")

    if paranoia:
        typer.echo("\n=== PARANOIA REPORT ===")
        typer.echo(f"Max |dE| (mm): {max_dE:.6f}")
        typer.echo(f"Max |dN| (mm): {max_dN:.6f}")
        typer.echo(f"Max |dH| (mm): {max_dH:.6f}")
        typer.echo(f"Worst point: {worst}")
        if max(max_dE, max_dN, max_dH) <= 0.5:
            typer.echo("✅ Paranoia check PASSED: mm-level integrity preserved.")
        else:
            raise RuntimeError("❌ Paranoia check FAILED: roundtrip exceeded 0.5 mm.")
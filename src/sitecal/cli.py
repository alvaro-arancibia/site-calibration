import typer
from pathlib import Path
from typing import Optional
from enum import Enum
import pandas as pd
from importlib import metadata

from sitecal.core.projections import ProjectionFactory
from sitecal.core.calibration_engine import CalibrationFactory
from sitecal.infrastructure.reports import generate_markdown_report
from sitecal.io import read_csv_to_dataframe

app = typer.Typer(no_args_is_help=True)

class CalibrationMethod(str, Enum):
    default = "default"
    ltm = "ltm"

@app.callback()
def main() -> None:
    """sitecal: Standard-compatible site calibration tools."""
    pass

@app.command()
def version() -> None:
    """Print version."""
    typer.echo(f"sitecal {metadata.version('sitecal')}")

@app.command()
def local2global(
    global_csv: Path = typer.Option(..., "--global-csv", exists=True, readable=True, help="CSV with global geodetic coordinates (Point,Lat,Lon,h)"),
    local_csv: Path = typer.Option(..., "--local-csv", exists=True, readable=True, help="CSV with local coordinates (Point,Easting,Northing,h_local)"),
    output_report: Path = typer.Option("calibration_report.md", help="Output report in Markdown format."),
    output_csv: Optional[Path] = typer.Option(None, help="Output CSV with transformed coordinates."),
    transform_csv: Optional[Path] = typer.Option(None, "--transform-csv", exists=True, readable=True, help="CSV with new GPS points (Point,Lat,Lon,h) to transform using the calibration."),
    method: CalibrationMethod = typer.Option("default", "--method", help="Calibration method"),
    # LTM parameters
    central_meridian: Optional[float] = typer.Option(None, help="LTM Central Meridian"),
    latitude_of_origin: Optional[float] = typer.Option(None, help="LTM Latitude of Origin"),
    false_easting: Optional[float] = typer.Option(None, help="LTM False Easting"),
    false_northing: Optional[float] = typer.Option(None, help="LTM False Northing"),
    scale_factor: Optional[float] = typer.Option(None, help="LTM Scale Factor"),

) -> None:
    """
    Performs a site calibration by projecting global coordinates and fitting them
    to a local plane coordinate system.
    """

    # Read data
    df_global = read_csv_to_dataframe(global_csv)
    df_local = read_csv_to_dataframe(local_csv)

    # Validate required LTM parameters if method is ltm
    if method.value == "ltm":
        ltm_params = [central_meridian, latitude_of_origin, false_easting, false_northing, scale_factor]
        if any(p is None for p in ltm_params):
            typer.echo("Error: For LTM method, all LTM parameters are required.", err=True)
            raise typer.Exit(code=1)

    # --- Projection Step ---
    projection_params = {
        "central_meridian": central_meridian,
        "latitude_of_origin": latitude_of_origin,
        "false_easting": false_easting,
        "false_northing": false_northing,
        "scale_factor": scale_factor,
    }
    projection = ProjectionFactory.create(method.value, **projection_params)
    df_global_proj = projection.project(df_global)

    # --- Calibration Step (2D Similarity) ---
    calibration = CalibrationFactory.create(method.value)
    calibration.train(df_local, df_global_proj)

    typer.echo("Calibration training completed.")

    # --- Reporting Step ---
    generate_markdown_report(calibration, output_report, method.value)
    typer.echo(f"Calibration report generated at: {output_report}")

    # --- Transformation & Output Step (control points) ---
    if output_csv and not transform_csv:
        df_to_transform = df_global_proj
        transformed_df = calibration.transform(df_to_transform)
        transformed_df.to_csv(output_csv, index=False)
        typer.echo(f"Transformed coordinates saved to: {output_csv}")

    # --- Transform New Points Step ---
    if transform_csv:
        df_new = read_csv_to_dataframe(transform_csv)
        df_new_proj = projection.project(df_new)
        transformed_new = calibration.transform(df_new_proj)
        out_path = output_csv if output_csv else Path("transformed_output.csv")
        transformed_new.to_csv(out_path, index=False)
        typer.echo(f"Transformed new points saved to: {out_path}")


if __name__ == "__main__":
    app()


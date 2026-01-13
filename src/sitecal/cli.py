import typer
from pathlib import Path
from typing import Optional
import pandas as pd

from sitecal.core.projections import ProjectionFactory
from sitecal.core.geocentric import geodetic_to_geocentric
from sitecal.core.calibration_engine import CalibrationFactory
from sitecal.infrastructure.reports import generate_markdown_report
from sitecal.io import read_csv_to_dataframe

app = typer.Typer(no_args_is_help=True)

@app.callback()
def main() -> None:
    """sitecal: TBC-compatible site calibration tools."""
    pass

@app.command()
def version() -> None:
    """Print version."""
    typer.echo("sitecal 0.2.0") # Increased version

@app.command()
def local2global(
    global_csv: Path = typer.Option(..., "--global-csv", exists=True, readable=True, help="CSV with global geodetic coordinates (Point,Lat,Lon,h)"),
    local_csv: Path = typer.Option(..., "--local-csv", exists=True, readable=True, help="CSV with local coordinates (Point,Easting,Northing,h_local)"),
    output_report: Path = typer.Option("calibration_report.md", help="Output report in Markdown format."),
    output_csv: Optional[Path] = typer.Option(None, help="Output CSV with transformed coordinates."),
    method: str = typer.Option("tbc", "--method", help="Calibration method: [tbc|helmert|ltm]"),
    # LTM parameters
    central_meridian: Optional[float] = typer.Option(None, help="LTM Central Meridian"),
    latitude_of_origin: Optional[float] = typer.Option(None, help="LTM Latitude of Origin"),
    false_easting: Optional[float] = typer.Option(None, help="LTM False Easting"),
    false_northing: Optional[float] = typer.Option(None, help="LTM False Northing"),
    scale_factor: Optional[float] = typer.Option(None, help="LTM Scale Factor"),

) -> None:
    """
    Performs a site calibration based on common control points between a global
    and a local coordinate system.
    """
    
    # Read data
    df_global = read_csv_to_dataframe(global_csv)
    df_local = read_csv_to_dataframe(local_csv)

    # Validate required LTM parameters if method is ltm
    if method == "ltm":
        ltm_params = [central_meridian, latitude_of_origin, false_easting, false_northing, scale_factor]
        if any(p is None for p in ltm_params):
            typer.echo("Error: For LTM method, all LTM parameters are required.", err=True)
            raise typer.Exit(code=1)

    # --- Calibration Step ---
    calibration = CalibrationFactory.create(method)

    if method == "helmert":
        # For Helmert, we need geocentric coordinates
        df_global_geo = geodetic_to_geocentric(df_global)
        
        # Assume E, N, H from local_csv are X, Y, Z in a local 3D system
        # The user said N=Y, E=X. The local file has columns N and E.
        # io.py maps N to Northing and E to Easting.
        # So we map Northing to Y and Easting to X.
        # Let's try swapping them: Northing -> X, Easting -> Y
        df_local_3d = df_local.rename(columns={"Northing": "X", "Easting": "Y", "h_local": "Z"})
        
        # The Helmert implementation expects the columns to be named X_local, Y_local, Z_local
        df_local_3d.rename(columns={"X": "X_local", "Y": "Y_local", "Z": "Z_local", 
                                      "Point": "Point"}, inplace=True)
        df_global_geo.rename(columns={"X": "X_global", "Y": "Y_global", "Z": "Z_global",
                                        "Point": "Point"}, inplace=True)

        calibration.train(df_local_3d, df_global_geo)

    else: # tbc or ltm (2D similarity)
        # --- Projection Step ---
        projection_params = {
            "central_meridian": central_meridian,
            "latitude_of_origin": latitude_of_origin,
            "false_easting": false_easting,
            "false_northing": false_northing,
            "scale_factor": scale_factor,
        }
        projection = ProjectionFactory.create(method, **projection_params)
        df_global_proj = projection.project(df_global)
        calibration.train(df_local, df_global_proj)
    
    typer.echo("Calibration training completed.")

    # --- Reporting Step ---
    generate_markdown_report(calibration, output_report, method)
    typer.echo(f"Calibration report generated at: {output_report}")
    
    # --- Transformation & Output Step ---
    if output_csv:
        if method == "helmert":
            df_to_transform = geodetic_to_geocentric(df_global)
        else:
            # This was defined inside the else block, so we need to redefine it here
            # to make it available in this scope.
            projection_params = {
                "central_meridian": central_meridian,
                "latitude_of_origin": latitude_of_origin,
                "false_easting": false_easting,
                "false_northing": false_northing,
                "scale_factor": scale_factor,
            }
            projection = ProjectionFactory.create(method, **projection_params)
            df_global_proj = projection.project(df_global)
            df_to_transform = df_global_proj
        
        transformed_df = calibration.transform(df_to_transform)
        transformed_df.to_csv(output_csv, index=False)
        typer.echo(f"Transformed coordinates saved to: {output_csv}")


if __name__ == "__main__":
    app()

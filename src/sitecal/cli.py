import typer

app = typer.Typer(no_args_is_help=True)

@app.callback()
def main() -> None:
    """sitecal: TBC-compatible site calibration tools."""
    pass

@app.command()
def version() -> None:
    """Print version."""
    typer.echo("sitecal 0.1.0")

if __name__ == "__main__":
    app()

from pathlib import Path
from sitecal.commands.local2global import run as _local2global_run

@app.command()
def local2global(
    cal_report: Path = typer.Option(..., exists=True, readable=True, help="TBC Site Calibration HTML report"),
    input_csv: Path = typer.Option(..., exists=True, readable=True, help="CSV with local coordinates (id,E,N,M)"),
    output_csv: Path = typer.Option(..., help="Output CSV (adds lon,lat,h)"),
    paranoia: bool = typer.Option(True, help="Run mm-level roundtrip integrity check"),
) -> None:
    """Convert local CSV -> global geodetic using a TBC calibration report."""
    _local2global_run(cal_report=cal_report, input_csv=input_csv, output_csv=output_csv, paranoia=paranoia)

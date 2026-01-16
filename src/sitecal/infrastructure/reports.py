import pandas as pd
from sitecal.core.calibration_engine import Calibration
import datetime
import pytz

def generate_markdown_report(
    calibration: Calibration,
    output_path: str,
    method: str
):
    """
    Generates a Markdown report with calibration results.
    """
    
    report_lines = []
    
    now = datetime.datetime.now(pytz.timezone('America/Santiago')).strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines.append("# Site Calibration Report")
    report_lines.append("")
    report_lines.append(f"Report generated on: {now}")
    report_lines.append("")
    report_lines.append(f"## Calibration Method: {method.upper()}")
    report_lines.append("")

    # Transformation Parameters
    report_lines.append("### Transformation Parameters")
    report_lines.append("")
    if calibration.params:
        params_df = pd.DataFrame([calibration.params])
        report_lines.append(params_df.to_markdown(index=False))
    else:
        report_lines.append("No parameters were calculated.")
    report_lines.append("")
    
    # Parameters Summary
    report_lines.append("### Calculated Parameters")
    report_lines.append("")
    if calibration.params:
        for key, value in calibration.params.items():
            report_lines.append(f"- **{key}:** `{value}`")
    else:
        report_lines.append("No parameters were calculated.")
    report_lines.append("")

    # Residuals Table
    report_lines.append("### Residuals (mm)")
    report_lines.append("")
    if calibration.residuals is not None:
        residuals_mm = calibration.residuals.copy()
        residuals_mm["dE"] = (residuals_mm["dE"] * 1000).round(1)
        residuals_mm["dN"] = (residuals_mm["dN"] * 1000).round(1)
        residuals_mm["dH"] = (residuals_mm["dH"] * 1000).round(1)
        residuals_mm.rename(columns={"dE": "dE (mm)", "dN": "dN (mm)", "dH": "dH (mm)"}, inplace=True)
        report_lines.append(residuals_mm.to_markdown(index=False))
    else:
        report_lines.append("No residuals were calculated.")
    report_lines.append("")

    # Statistics
    report_lines.append("### Statistics")
    report_lines.append("")
    if calibration.residuals is not None:
        residuals = calibration.residuals
        # Calculate horizontal error
        residuals["error_h"] = (residuals["dE"]**2 + residuals["dN"]**2)**0.5
        
        worst_point = residuals.loc[residuals["error_h"].idxmax()]
        best_point = residuals.loc[residuals["error_h"].idxmin()]
        std_dev = residuals[["dE", "dN", "dH"]].std().to_dict()
        percentile_99 = residuals["error_h"].quantile(0.99)

        report_lines.append(f"- **Worst Point:** `{worst_point['Point']}` (Error: {(worst_point['error_h'] * 1000):.1f} mm)")
        report_lines.append(f"- **Best Point:** `{best_point['Point']}` (Error: {(best_point['error_h'] * 1000):.1f} mm)")
        report_lines.append("- **Standard Deviations (mm):**")
        for axis, value in std_dev.items():
            report_lines.append(f"  - `{axis}`: {(value * 1000):.1f} mm")
        report_lines.append(f"- **99th Percentile of Horizontal Errors:** {(percentile_99 * 1000):.1f} mm")

    else:
        report_lines.append("Statistics could not be calculated.")
    report_lines.append("")

    return "\n".join(report_lines) + "\n"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import numpy as np
import typing
import tempfile
import os

from sitecal.io import read_csv_to_dataframe
from sitecal.core.calibration_engine import CalibrationFactory
from sitecal.core.projections import ProjectionFactory
from sitecal.infrastructure.reports import generate_markdown_report
from sitecal.api.schemas import CalibrationResult, CalibrationParameters, ResidualPoint

app = FastAPI(title="Site Calibration API")

def validate_collinearity(df: pd.DataFrame) -> bool:
    """
    Checks if points are collinear by examining the variance in a 2D plane.
    If they are nearly in a line, the transformation might be unstable.
    """
    if "Easting_global" not in df.columns or "Northing_global" not in df.columns:
        return False
        
    coords = df[["Easting_global", "Northing_global"]].values
    if len(coords) < 3:
        return False
    
    # Simple collinearity check using PCA-like approach (eigenvalues of covariance matrix)
    centered = coords - np.mean(coords, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvals(cov)
    
    # If the ratio of the smallest to largest eigenvalue is very small, they are collinear
    if np.max(eigvals) == 0:
        return True
    
    ratio = np.min(eigvals) / np.max(eigvals)
    return ratio < 1e-4  # Threshold for collinearity

@app.post("/calibrate", response_model=CalibrationResult)
async def calibrate(
    method: str = Form("tbc"),
    local_csv: UploadFile = File(...),
    global_csv: UploadFile = File(...),
    central_meridian: typing.Optional[float] = Form(None),
    latitude_of_origin: typing.Optional[float] = Form(None),
    false_easting: typing.Optional[float] = Form(None),
    false_northing: typing.Optional[float] = Form(None),
    scale_factor: typing.Optional[float] = Form(None)
):
    try:
        # Read CSVs
        local_content = await local_csv.read()
        global_content = await global_csv.read()
        
        # We need to normalize them using our io logic
        # Since read_csv_to_dataframe expects a path, we'll use a temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode='wb', delete=False) as tmp_local, \
             tempfile.NamedTemporaryFile(suffix=".csv", mode='wb', delete=False) as tmp_global:
            tmp_local.write(local_content)
            tmp_local.close()
            tmp_global.write(global_content)
            tmp_global.close()
            
            try:
                df_l = read_csv_to_dataframe(tmp_local.name)
                df_g = read_csv_to_dataframe(tmp_global.name)
            finally:
                if os.path.exists(tmp_local.name): os.remove(tmp_local.name)
                if os.path.exists(tmp_global.name): os.remove(tmp_global.name)

        # --- Projection Step ---
        projection_params = {
            "central_meridian": central_meridian,
            "latitude_of_origin": latitude_of_origin,
            "false_easting": false_easting,
            "false_northing": false_northing,
            "scale_factor": scale_factor,
        }
        
        # Validate LTM params if needed
        if method == "ltm" and any(v is None for v in projection_params.values()):
            raise HTTPException(status_code=400, detail="For LTM method, all projection parameters are required.")
            
        projection = ProjectionFactory.create(method, **projection_params)
        df_g_proj = projection.project(df_g)

        # Merge to find common points
        merged_df = pd.merge(df_l, df_g_proj, on="Point", suffixes=('_local', '_global'))
        
        # Validation 1: At least 3 common points
        if len(merged_df) < 3:
            raise HTTPException(status_code=400, detail=f"Found only {len(merged_df)} common points. Minimum 3 are required.")
            
        # Validation 2: Collinearity check
        if validate_collinearity(merged_df):
            raise HTTPException(status_code=400, detail="Points are collinear or too close to each other, preventing a stable transformation.")

        # Train model
        engine = CalibrationFactory.create(method)
        engine.train(df_l, df_g_proj)
        
        # Generate Report
        report_text = generate_markdown_report(engine, "not_needed_anymore", method)
        
        # Prepare response
        params = CalibrationParameters(**engine.params)
        residuals = [
            ResidualPoint(
                Point=str(row["Point"]), 
                dE=float(row["dE"]), 
                dN=float(row["dN"]), 
                dH=float(row["dH"])
            ) 
            for _, row in engine.residuals.iterrows()
        ]
        
        return CalibrationResult(
            parameters=params,
            residuals=residuals,
            report=report_text
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

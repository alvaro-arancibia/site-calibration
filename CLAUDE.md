# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sitecal` is a site calibration tool for transforming global geodetic coordinates (WGS84) to local plane coordinate systems. It provides both a CLI and a Streamlit web UI for performing industry-standard coordinate transformations.

**Key capabilities:**
- Project geodetic coordinates (Lat/Lon) to planar systems using Default or LTM (Local Transverse Mercator) methods
- Calculate 2D similarity transformation parameters (translation, rotation, scale)
- Apply vertical adjustment with inclined plane fitting
- Generate calibration reports with residuals and statistics

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Unix/MacOS)
source .venv/bin/activate

# Install package in development mode
pip install -e .
```

### Running the Application
```bash
# CLI - Basic calibration
sitecal local2global --global-csv <path> --local-csv <path> --output-report report.md

# CLI - LTM method with parameters
sitecal local2global --global-csv <path> --local-csv <path> --method ltm \
  --central-meridian -70.5 --latitude-of-origin -33.4 \
  --false-easting 500000 --false-northing 10000000 --scale-factor 1.0

# Streamlit UI
streamlit run src/sitecal/ui/app.py
```

### Other Commands
```bash
# Check version
sitecal version
```

## Architecture

### Core Processing Pipeline

The calibration workflow follows this sequence:

1. **Input CSV Files** → `io.py` reads and validates data
2. **Projection** → `core/projections.py` converts geodetic coords to planar
3. **Calibration** → `core/calibration_engine.py` computes transformation parameters
4. **Output** → `infrastructure/reports.py` generates markdown reports

### Key Components

**Projection System** (`core/projections.py`)
- `ProjectionFactory`: Creates projection instances based on method
- `Default`: Uses first point as origin with TM projection (scale=1.0)
- `LTM`: Custom Transverse Mercator with user-defined parameters
- All projections use pyproj for coordinate transformation

**Calibration Engine** (`core/calibration_engine.py`)
- `Similarity2D`: Main calibration class implementing 2D similarity transformation
- Horizontal adjustment: 4 parameters (a, b, tE, tN) via least squares
- Vertical adjustment: Inclined plane model (constant + slope_N + slope_E)
- Uses centered coordinates for numerical stability

**Data Requirements:**
- Global CSV: `Point`, `Latitude`, `Longitude`, `EllipsoidalHeight`
- Local CSV: `Point`, `Easting`, `Northing`, `Elevation`
- Point IDs must match between files (minimum 3 common points)

### Code Organization

```
src/sitecal/
├── cli.py                      # Typer CLI entry point
├── io.py                       # CSV reading and validation
├── angles.py                   # Angle utility functions
├── core/
│   ├── calibration_engine.py  # Similarity2D transformation
│   └── projections.py         # Projection factory and implementations
├── infrastructure/
│   └── reports.py             # Markdown report generation
└── ui/
    └── app.py                 # Streamlit web interface
```

### Important Implementation Details

**Calibration Mathematics:**
- Horizontal parameters solved using centered coordinates to avoid numerical instability
- Centroid calculation: `x_c = mean(x)`, then work with `x_prime = x - x_c`
- Transformation equations:
  - `E_local = a * E_global - b * N_global + tE`
  - `N_local = b * E_global + a * N_global + tN`
- Vertical error model: `Z_error = C + S_N * (N - N_c) + S_E * (E - E_c)`

**Column Name Strictness:**
- Core engine expects exact column names (no fuzzy matching in core modules)
- UI layer (`app.py`) handles user column mapping
- Always convert `Point` to string type after reading CSV

**Collinearity Check:**
- UI validates point geometry to prevent unstable calculations
- Check eigenvalue ratio of covariance matrix: `min(eigvals) / max(eigvals) < 1e-4`

## Testing Calibration Changes

When modifying calibration logic:
1. Test with minimum viable dataset (3 points, non-collinear)
2. Verify both horizontal and vertical parameter calculation
3. Check residuals are computed correctly
4. Ensure report generation works with calculated parameters
5. Test edge cases: collinear points, insufficient points, missing columns

## GitHub Pages Deployment

The PWA lives in `docs/` directory for GitHub Pages compatibility. The `docs/index.html` is a standalone Progressive Web App that can run offline.

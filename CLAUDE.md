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

**Python CLI:**
```bash
# Basic calibration
sitecal local2global --global-csv <path> --local-csv <path> --output-report report.md

# LTM method with parameters
sitecal local2global --global-csv <path> --local-csv <path> --method ltm \
  --central-meridian -70.5 --latitude-of-origin -33.4 \
  --false-easting 500000 --false-northing 10000000 --scale-factor 1.0
```

**Python Streamlit UI:**
```bash
streamlit run src/sitecal/ui/app.py
```

**JavaScript Standalone (No installation needed):**
```bash
# Simply open in browser
open index.html
# or
python -m http.server 8000  # then navigate to http://localhost:8000
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

### JavaScript Implementation Details

The `index.html` file contains a complete reimplementation in JavaScript:

**Libraries Used:**
- `proj4js` - Coordinate transformations (equivalent to Python's pyproj)
- `mathjs` - Matrix operations and least squares solving (equivalent to numpy)
- `papaparse` - CSV parsing (equivalent to pandas.read_csv)
- `chart.js` - Visualization

**Key Differences from Python:**
- Matrix operations use math.js `lusolve()` instead of numpy's `lstsq()`
- Proj4 requires explicit projection string definitions (same format as pyproj)
- Eigenvalue calculation for collinearity check is manual (2x2 matrix formula)
- All data manipulation uses plain JavaScript arrays instead of pandas DataFrames

**Architecture Similarity:**
- `ProjectionEngine` class ≈ `projections.py` module
- `CalibrationEngine` class ≈ `Similarity2D` class in `calibration_engine.py`
- Same mathematical formulas and centered coordinate approach
- Identical parameter calculation (a, b, tE, tN, C, slope_n, slope_e)

## Testing Calibration Changes

When modifying calibration logic:
1. Test with minimum viable dataset (3 points, non-collinear)
2. Verify both horizontal and vertical parameter calculation
3. Check residuals are computed correctly
4. Ensure report generation works with calculated parameters
5. Test edge cases: collinear points, insufficient points, missing columns

## Implementations

This project has two implementations:

### 1. Python Implementation (Primary)
Located in `src/sitecal/`:
- CLI tool via Typer (`sitecal` command)
- Streamlit web UI (`src/sitecal/ui/app.py`)
- Core calibration engine with numpy/pandas
- Uses pyproj for coordinate transformations

### 2. JavaScript Implementation (Standalone)
Located in `index.html` at project root:
- Single-file HTML application (100% offline capable)
- Pure JavaScript with CDN libraries:
  - Bootstrap 5 for UI
  - Papa Parse for CSV parsing
  - Proj4js for coordinate projections
  - Chart.js for geometry visualization
  - Math.js for linear algebra
- Complete calibration engine reimplemented in JavaScript
- Responsive dark theme interface
- No build step required - just open in browser

**Key features of JS implementation:**
- Real-time CSV preview and column mapping
- Interactive geometry chart for local points
- Support for Default, LTM, and UTM projections
- Full 2D similarity transformation + vertical adjustment
- Collinearity detection
- Comprehensive results display with statistics
- Markdown report generation with copy functionality

### 3. GitHub Pages PWA
Located in `docs/` directory:
- Uses stlite (Streamlit compiled to WebAssembly)
- Runs Python code in browser
- Heavier but provides full Python Streamlit experience

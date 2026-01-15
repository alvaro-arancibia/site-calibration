# sitecal

`sitecal` is a Python-based CLI tool for performing site calibrations and coordinate transformations compatible with Trimble Business Center (TBC).

## Key Features

- **TBC Compatible**: Emulates TBC's default calibration behavior.
- **WGS84 Focus**: All transformations are performed using the WGS84 geodetic datum.
- **Support for Local Projections**:
  - TBC Default (Local Transverse Mercator).
  - UTM (Universal Transverse Mercator) with auto-zone detection.
  - Custom LTM (Local Transverse Mercator) with user-defined parameters.
- **Transformation Engines**:
  - 2D Similarity (4 parameters) for fitting projected coordinates to a local plane.
- **Reporting**: Generates detailed Markdown reports including parameters and residual analysis.

## Installation

### Prerequisites

- Python 3.11 or higher.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/alvaro-arancibia/site-calibration.git
   cd site-calibration
   ```

2. Create a virtual environment and install the package:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install .
   ```

## Quick Start

Perform a basic site calibration using the TBC default method:

```bash
sitecal local2global \
  --global-csv path/to/global_coords.csv \
  --local-csv path/to/local_coords.csv \
  --method tbc \
  --output-report report.md
```

For more detailed information on methods and parameters, see the [Calibration Documentation](docs/calibration.md).

## Project Structure

- `src/sitecal/`: Main package source code.
- `docs/`: Detailed technical documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
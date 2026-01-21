# Site Calibration Pro (`sitecal`)

**Site Calibration Pro** is a modern, offline-first tool for performing site calibrations and coordinate transformations. It is compatible with industry standards and runs entirely in your browser as a **Progressive Web App (PWA)** or via the command line (CLI).

## Key Features

- **Offline-First PWA**: Runs 100% in the browser (Tablet/Desktop) using WebAssembly (Pyodide). No backend server required.
- **Industry Standard Compatible**: Emulates standard default calibration behavior (formerly specific to legacy software).
- **Secure & Private**: All processing happens locally on your device. Your data never leaves your browser.
- **Transformation Engines**:
  - 2D Similarity (4-parameter) adjustments.
  - Vertical Adjustment (Inclined Plane or Constant Shift).
- **Projections**:
  - Default (Local Transverse Mercator).
  - UTM (Universal Transverse Mercator) with auto-zone detection.
  - Custom LTM (Local Transverse Mercator).

---

## 🚀 Web / Tablet Usage (PWA)

The easiest way to use Site Calibration Pro is via the Web Interface.

1. **Access the App**: Navigate to the hosted version (e.g., via GitHub Pages) or run it locally.
2. **Install on Tablet (iPad / Android)**:
    - Tap "Share" (iOS) or the Menu (Android).
    - Select **"Add to Home Screen"**.
    - The app will now work locally, **even without an internet connection**.

### Running Locally (Browser)

If you have the source code and want to run the PWA locally:

```bash
cd dist
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## 💻 CLI Usage (Python)

For automation or desktop usage, you can use the Command Line Interface.

### Installation

```bash
git clone https://github.com/alvaro-arancibia/site-calibration.git
cd site-calibration
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Quick Start (CLI)

Perform a basic site calibration:

```bash
sitecal local2global \
  --global-csv data/global_points.csv \
  --local-csv data/local_points.csv \
  --method default \
  --output-report report.md
```

For more detailed information on methods and parameters, see the [Calibration Documentation](docs/calibration.md).

---

## Project Structure

- `dist/`: **Production PWA**. Contains `index.html` (monolithic) and assets. Ready for static hosting.
- `src/sitecal/`: Core Python package (shared by CLI and PWA).
- `docs/`: Technical documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

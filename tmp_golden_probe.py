"""
Probe script: runs the calibration engine with a known reference dataset
and prints the exact output values to be used as golden references in tests.

Run with:
    python tmp_golden_probe.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sitecal.core.calibration_engine import Similarity2D

# -----------------------------------------------------------------------
# Reference dataset
# We build a dataset where the true horizontal transformation is:
#   E_local = a_true * x_global - b_true * y_global + tE_true
#   N_local = b_true * x_global + a_true * y_global + tN_true
# and the vertical error follows an inclined plane:
#   Z_error = C_true + Sn_true*(N_local - Nc) + Se_true*(E_local - Ec)
# so that h_local = h_global - Z_error
# -----------------------------------------------------------------------

a_true  =  0.9999
b_true  =  0.0050
tE_true =  500.0
tN_true =  300.0

C_true  =  0.300
Sn_true =  0.0001
Se_true = -0.0002

x_global = np.array([1000.0, 2000.0, 1500.0, 2500.0])
y_global = np.array([1000.0, 1000.0, 2000.0, 2000.0])
h_global = np.array([100.50, 101.00,  99.80, 100.20])

E_local = a_true * x_global - b_true * y_global + tE_true
N_local = b_true * x_global + a_true * y_global + tN_true

Nc = np.mean(N_local)
Ec = np.mean(E_local)

Z_err   = C_true + Sn_true * (N_local - Nc) + Se_true * (E_local - Ec)
h_local = h_global - Z_err

# DataFrames expected by the engine
df_global = pd.DataFrame({
    "Point":             ["P1", "P2", "P3", "P4"],
    "Easting":           x_global,
    "Northing":          y_global,
    "EllipsoidalHeight": h_global,
})

df_local = pd.DataFrame({
    "Point":     ["P1", "P2", "P3", "P4"],
    "Easting":   E_local,
    "Northing":  N_local,
    "Elevation": h_local,
})

# -----------------------------------------------------------------------
# Train the model
# -----------------------------------------------------------------------
model = Similarity2D()
model.train(df_local, df_global)

print("=" * 60)
print("HORIZONTAL PARAMETERS")
print("=" * 60)
hp = model.horizontal_params
for k, v in hp.items():
    print(f"  {k:20s} = {v!r}")

print()
print("=" * 60)
print("VERTICAL PARAMETERS")
print("=" * 60)
vp = model.vertical_params
for k, v in vp.items():
    print(f"  {k:20s} = {v!r}")

print()
print("=" * 60)
print("TRANSFORM OUTPUT  (global -> local)")
print("=" * 60)
result = model.transform(df_global)
print(result.to_string(index=False))
print()
print("Raw arrays:")
print("  Easting  =", repr(result["Easting"].values))
print("  Northing =", repr(result["Northing"].values))
print("  h        =", repr(result["h"].values))

print()
print("=" * 60)
print("RESIDUALS (should be near-zero for perfect data)")
print("=" * 60)
print(model.residuals.to_string(index=False))
print()
print("dE =", repr(model.residuals["dE"].values))
print("dN =", repr(model.residuals["dN"].values))
print("dH =", repr(model.residuals["dH"].values))

# -----------------------------------------------------------------------
# Extra: test untrained-model guard
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("GUARD TEST (untrained model should raise RuntimeError)")
print("=" * 60)
try:
    Similarity2D().transform(df_global)
    print("ERROR: expected RuntimeError was NOT raised")
except RuntimeError as e:
    print(f"  OK: raised RuntimeError: {e}")

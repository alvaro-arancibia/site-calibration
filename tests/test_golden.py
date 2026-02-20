"""
tests/test_golden.py
====================
Golden-State regression tests for the Site Calibration Pro engine.

PURPOSE
-------
These tests fix the *current* observable behavior of the calibration engine
so that any future refactoring can be verified to preserve it exactly.

They were written in Phase 0, before any code changes, by:
1. Choosing a synthetic reference dataset with a KNOWN true transformation.
2. Running the algebra of the engine by hand to derive the expected outputs.
3. Hardcoding those values here with np.testing.assert_allclose(rtol=1e-5).

If a test breaks after a refactor, the engine's *numerical contract* has
changed and must be explicitly re-approved by the team before accepting.

REFERENCE DATASET
-----------------
True horizontal transformation (Similarity2D):
    a  = 0.9999   (scale * cos(θ))
    b  = 0.0050   (scale * sin(θ))
    tE = 500.0    (easting  translation)
    tN = 300.0    (northing translation)

    E_local = a*x_global - b*y_global + tE
    N_local = b*x_global + a*y_global + tN

True vertical adjustment (Inclined Plane):
    C  = 0.300    (constant shift, m)
    Sn = 0.0001   (slope in northing direction, m/m)
    Se = -0.0002  (slope in easting direction,  m/m)

    Z_error = C + Sn*(N_local - Nc) + Se*(E_local - Ec)
    h_local = h_global - Z_error

Control points:
    P1: x=1000, y=1000, h=100.50
    P2: x=2000, y=1000, h=101.00
    P3: x=1500, y=2000, h= 99.80
    P4: x=2500, y=2000, h=100.20

All expected values below were computed analytically from the above
definition. With noise-free data the engine must recover the true
parameters exactly (to machine-precision, well within rtol=1e-5).
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the source tree importable when pytest is run from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sitecal.core.calibration_engine import Similarity2D, CalibrationFactory


# ===========================================================================
# Shared fixture: reference dataset + trained model
# ===========================================================================

# --- True horizontal parameters ---
A_TRUE  =  0.9999
B_TRUE  =  0.0050
TE_TRUE =  500.0
TN_TRUE =  300.0

# --- True vertical parameters ---
C_TRUE  =  0.300
SN_TRUE =  0.0001
SE_TRUE = -0.0002

# --- Global coordinates ---
X_GLOBAL = np.array([1000.0, 2000.0, 1500.0, 2500.0])
Y_GLOBAL = np.array([1000.0, 1000.0, 2000.0, 2000.0])
H_GLOBAL = np.array([100.50, 101.00,  99.80, 100.20])

# --- Local coordinates derived analytically ---
E_LOCAL = A_TRUE * X_GLOBAL - B_TRUE * Y_GLOBAL + TE_TRUE
# P1=1494.9, P2=2494.8, P3=1989.85, P4=2989.75

N_LOCAL = B_TRUE * X_GLOBAL + A_TRUE * Y_GLOBAL + TN_TRUE
# P1=1304.9, P2=1309.9, P3=2307.3, P4=2312.3

# Centroids of local coordinates
EC = np.mean(E_LOCAL)   # 2242.325
NC = np.mean(N_LOCAL)   # 1808.6

# Z_error at each control point (from the inclined-plane model)
Z_ERROR = C_TRUE + SN_TRUE * (N_LOCAL - NC) + SE_TRUE * (E_LOCAL - EC)
# P1≈0.399115, P2≈0.199635, P3≈0.400365, P4≈0.200885

H_LOCAL = H_GLOBAL - Z_ERROR
# P1≈100.100885, P2≈100.800365, P3≈99.399635, P4≈99.999115

POINT_IDS = ["P1", "P2", "P3", "P4"]


@pytest.fixture(scope="module")
def reference_dfs():
    """Return (df_local, df_global) DataFrames for the reference dataset."""
    df_global = pd.DataFrame({
        "Point":             POINT_IDS,
        "Easting":           X_GLOBAL,
        "Northing":          Y_GLOBAL,
        "EllipsoidalHeight": H_GLOBAL,
    })
    df_local = pd.DataFrame({
        "Point":     POINT_IDS,
        "Easting":   E_LOCAL,
        "Northing":  N_LOCAL,
        "Elevation": H_LOCAL,
    })
    return df_local, df_global


@pytest.fixture(scope="module")
def trained_model(reference_dfs):
    """Return a Similarity2D model trained on the reference dataset."""
    df_local, df_global = reference_dfs
    model = Similarity2D()
    model.train(df_local, df_global)
    return model


# ===========================================================================
# 1. HORIZONTAL ADJUSTMENT — 2-D Similarity parameters
# ===========================================================================

class TestHorizontalAdjustment:
    """Pin the exact fitted values of a, b, tE, tN."""

    def test_parameter_a_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.horizontal_params["a"],
            A_TRUE,
            rtol=1e-5,
            err_msg="Horizontal parameter 'a' (scale*cos) has changed",
        )

    def test_parameter_b_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.horizontal_params["b"],
            B_TRUE,
            rtol=1e-5,
            err_msg="Horizontal parameter 'b' (scale*sin) has changed",
        )

    def test_centroid_easting_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.horizontal_params["E_c"],
            EC,
            rtol=1e-5,
            err_msg="Horizontal centroid E_c has changed",
        )

    def test_centroid_northing_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.horizontal_params["N_c"],
            NC,
            rtol=1e-5,
            err_msg="Horizontal centroid N_c has changed",
        )

    def test_horizontal_params_dict_has_required_keys(self, trained_model):
        required = {"a", "b", "E_c", "N_c", "x_c", "y_c"}
        assert required.issubset(trained_model.horizontal_params.keys()), (
            f"horizontal_params is missing keys: "
            f"{required - set(trained_model.horizontal_params.keys())}"
        )


# ===========================================================================
# 2. VERTICAL ADJUSTMENT — Inclined Plane parameters
# ===========================================================================

class TestVerticalAdjustment:
    """Pin the exact fitted values of the inclined-plane model."""

    def test_vertical_shift_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.vertical_params["vertical_shift"],
            C_TRUE,
            rtol=1e-5,
            err_msg="Vertical shift C has changed",
        )

    def test_slope_north_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.vertical_params["slope_north"],
            SN_TRUE,
            rtol=1e-5,
            err_msg="Vertical slope_north Sn has changed",
        )

    def test_slope_east_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.vertical_params["slope_east"],
            SE_TRUE,
            rtol=1e-5,
            err_msg="Vertical slope_east Se has changed",
        )

    def test_centroid_north_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.vertical_params["centroid_north"],
            NC,
            rtol=1e-5,
            err_msg="centroid_north has changed",
        )

    def test_centroid_east_is_exact(self, trained_model):
        np.testing.assert_allclose(
            trained_model.vertical_params["centroid_east"],
            EC,
            rtol=1e-5,
            err_msg="centroid_east has changed",
        )

    def test_vertical_params_dict_has_required_keys(self, trained_model):
        required = {"vertical_shift", "slope_north", "slope_east",
                    "centroid_north", "centroid_east"}
        assert required.issubset(trained_model.vertical_params.keys()), (
            f"vertical_params is missing keys: "
            f"{required - set(trained_model.vertical_params.keys())}"
        )


# ===========================================================================
# 3. BIDIRECTIONAL TRANSFORM — Global → Local
# ===========================================================================

class TestTransformGlobalToLocal:
    """
    Pin the exact numerical output of Similarity2D.transform() when fed
    global coordinates. With noise-free data the transformed values must
    equal the original local coordinates to within rtol=1e-5.
    """

    @pytest.fixture(scope="class")
    def transform_result(self, trained_model, reference_dfs):
        _, df_global = reference_dfs
        return trained_model.transform(df_global)

    def test_output_easting_matches_local(self, transform_result):
        np.testing.assert_allclose(
            transform_result["Easting"].values,
            E_LOCAL,
            rtol=1e-5,
            err_msg="transform() Easting output has changed",
        )

    def test_output_northing_matches_local(self, transform_result):
        np.testing.assert_allclose(
            transform_result["Northing"].values,
            N_LOCAL,
            rtol=1e-5,
            err_msg="transform() Northing output has changed",
        )

    def test_output_height_matches_local(self, transform_result):
        np.testing.assert_allclose(
            transform_result["h"].values,
            H_LOCAL,
            rtol=1e-5,
            err_msg="transform() height output has changed",
        )

    def test_output_point_ids_preserved(self, transform_result):
        assert list(transform_result["Point"]) == POINT_IDS, (
            "transform() must preserve Point identifiers"
        )

    def test_output_dataframe_has_required_columns(self, transform_result):
        required = {"Point", "Easting", "Northing", "h"}
        assert required.issubset(transform_result.columns), (
            f"transform() output is missing columns: "
            f"{required - set(transform_result.columns)}"
        )


# ===========================================================================
# 4. RESIDUALS — near-zero for noise-free data
# ===========================================================================

class TestResiduals:
    """
    With a perfectly consistent (noise-free) dataset the residuals stored
    on the model after training must be at or near machine zero.
    """

    def test_residual_dE_near_zero(self, trained_model):
        np.testing.assert_allclose(
            trained_model.residuals["dE"].values,
            np.zeros(4),
            atol=1e-8,
            err_msg="Horizontal residuals dE are not near zero for perfect data",
        )

    def test_residual_dN_near_zero(self, trained_model):
        np.testing.assert_allclose(
            trained_model.residuals["dN"].values,
            np.zeros(4),
            atol=1e-8,
            err_msg="Horizontal residuals dN are not near zero for perfect data",
        )

    def test_residual_dH_near_zero(self, trained_model):
        np.testing.assert_allclose(
            trained_model.residuals["dH"].values,
            np.zeros(4),
            atol=1e-8,
            err_msg="Vertical residuals dH are not near zero for perfect data",
        )


# ===========================================================================
# 5. GUARD RAILS — API contracts
# ===========================================================================

class TestApiContracts:
    """Engine must enforce its public API contracts."""

    def test_transform_before_train_raises_runtime_error(self):
        """Calling transform() on an untrained model must raise RuntimeError."""
        model = Similarity2D()
        df_dummy = pd.DataFrame({
            "Point":   ["P1"],
            "Easting": [1000.0],
            "Northing":[1000.0],
            "EllipsoidalHeight": [100.0],
        })
        with pytest.raises(RuntimeError, match="not been trained"):
            model.transform(df_dummy)

    def test_factory_creates_similarity2d_for_default(self):
        """CalibrationFactory('default') must return a Similarity2D instance."""
        model = CalibrationFactory.create("default")
        assert isinstance(model, Similarity2D)

    def test_factory_creates_similarity2d_for_ltm(self):
        """CalibrationFactory('ltm') must return a Similarity2D instance."""
        model = CalibrationFactory.create("ltm")
        assert isinstance(model, Similarity2D)

    def test_factory_raises_for_unknown_method(self):
        """CalibrationFactory must raise ValueError for unknown methods."""
        with pytest.raises(ValueError, match="Unknown calibration method"):
            CalibrationFactory.create("nonexistent_method")

    def test_trained_model_has_horizontal_params(self, trained_model):
        assert trained_model.horizontal_params is not None

    def test_trained_model_has_vertical_params(self, trained_model):
        assert trained_model.vertical_params is not None

    def test_trained_model_has_residuals(self, trained_model):
        assert trained_model.residuals is not None


# ===========================================================================
# 6. VERTICAL-ONLY — Constant-shift fallback with < 3 points
# ===========================================================================

class TestConstantShiftFallback:
    """
    With only 2 control points, the engine must fall back to a constant
    vertical shift (slope_north = slope_east = 0) instead of a plane fit.
    """

    @pytest.fixture(scope="class")
    def model_2pts(self):
        """Train a model with exactly 2 control points."""
        # Use the first two points of the reference dataset
        df_global = pd.DataFrame({
            "Point":             ["P1", "P2"],
            "Easting":           X_GLOBAL[:2],
            "Northing":          Y_GLOBAL[:2],
            "EllipsoidalHeight": H_GLOBAL[:2],
        })
        df_local = pd.DataFrame({
            "Point":     ["P1", "P2"],
            "Easting":   E_LOCAL[:2],
            "Northing":  N_LOCAL[:2],
            "Elevation": H_LOCAL[:2],
        })
        model = Similarity2D()
        model.train(df_local, df_global)
        return model

    def test_slope_north_is_zero_for_2pts(self, model_2pts):
        assert model_2pts.vertical_params["slope_north"] == 0.0, (
            "With 2 points, slope_north must be 0 (fallback to constant shift)"
        )

    def test_slope_east_is_zero_for_2pts(self, model_2pts):
        assert model_2pts.vertical_params["slope_east"] == 0.0, (
            "With 2 points, slope_east must be 0 (fallback to constant shift)"
        )

    def test_constant_shift_is_mean_z_error_for_2pts(self, model_2pts):
        """C must equal the mean Z_error of the 2-point subset."""
        expected_shift = np.mean(Z_ERROR[:2])
        np.testing.assert_allclose(
            model_2pts.vertical_params["vertical_shift"],
            expected_shift,
            rtol=1e-5,
            err_msg="2-point fallback constant shift is wrong",
        )

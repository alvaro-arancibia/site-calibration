import json
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sitecal.domain.schemas import PointLocal, PointGlobal, PointTransform, HorizontalParams, VerticalParams
from sitecal.core.math_engine import calculate_wls_similarity, calculate_wls_vertical, check_extrapolation


class Calibration(ABC):
    @abstractmethod
    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self) -> str:
        """Serializes the calibrated model to a JSON string."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, json_str: str) -> 'Calibration':
        """Deserializes a JSON string into a Calibration model."""
        pass


class Similarity2D(Calibration):
    def __init__(self):
        self.horizontal_params = None
        self.vertical_params = None
        self.residuals = None

    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        """
        Calculates 2D similarity transformation parameters (a, b, tE, tN)
        and Vertical Adjustment parameters (Inclined Plane or Constant Shift).
        """
        
        # Pydantic validation for input DataFrames
        local_pts = [PointLocal(**row) for row in df_local.to_dict('records')]
        global_pts = [PointGlobal(**row) for row in df_global.to_dict('records')]

        merged_df = pd.merge(df_local, df_global, on="Point", suffixes=('_local', '_global'))
        n = len(merged_df)

        # Extract values
        x = merged_df["Easting_global"].values
        y = merged_df["Northing_global"].values
        E = merged_df["Easting_local"].values
        N = merged_df["Northing_local"].values
        
        # Determine weights
        sigma_col = "sigma_global" if "sigma_global" in merged_df.columns else "sigma"
        if sigma_col in merged_df.columns:
            sigmas = merged_df[sigma_col].values
            sigmas_safe = np.where(sigmas > 0, sigmas, 1.0)
            w_sqrt = 1.0 / sigmas_safe
        else:
            w_sqrt = np.ones(n)
            
        w_sq = w_sqrt ** 2
        sum_w = float(np.sum(w_sq))

        # --- Horizontal Adjustment ---
        a, b, x_c, y_c, E_c, N_c, outlier_flags_h, sigma0_sq_h, local_control_pts = calculate_wls_similarity(
            x, y, E, N, w_sq, sum_w, w_sqrt, n
        )
        
        tE = E_c - a * x_c + b * y_c
        tN = N_c - b * x_c - a * y_c
        
        self.horizontal_params = HorizontalParams(
            a=a, b=b, x_c=x_c, y_c=y_c, E_c=E_c, N_c=N_c,
            tE=tE, tN=tN,
            local_control_points=local_control_pts.tolist()
        ).model_dump()

        # --- Vertical Adjustment ---
        h_global = merged_df["EllipsoidalHeight"].values 
        h_local = merged_df["Elevation"].values
        Z_error = h_global - h_local
        
        # We need N_prime and E_prime for the vertical planar fit domain
        E_prime = E - E_c
        N_prime = N - N_c
        
        C, slope_n, slope_e, rank, bad_geom, bad_cond, outlier_flags_v, sigma0_sq_v = calculate_wls_vertical(
            Z_error, E_prime, N_prime, w_sq, sum_w, w_sqrt, n, N_c, E_c
        )
            
        self.vertical_params = VerticalParams(
            vertical_shift=C,
            slope_north=slope_n,
            slope_east=slope_e,
            centroid_north=N_c,
            centroid_east=E_c,
            rank=rank,
            bad_geometry=bad_geom,
            bad_condition=bad_cond
        ).model_dump()
        
        # Calculate residuals & Transformed Values
        transformed = self.transform(merged_df)
        self.residuals = pd.DataFrame({
            "Point": merged_df["Point"],
            "dE": transformed["Easting"] - merged_df["Easting_local"],
            "dN": transformed["Northing"] - merged_df["Northing_local"],
            "dH": transformed["h"] - merged_df.get("Elevation", 0) - (merged_df.get("EllipsoidalHeight", 0) - merged_df.get("Elevation", 0)), 
            "outlier_horizontal": outlier_flags_h,
            "outlier_vertical": outlier_flags_v
        })
        self.residuals["dH"] = transformed["h"] - merged_df["Elevation"]


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.horizontal_params is None or self.vertical_params is None:
            raise RuntimeError("The calibration model has not been trained.")
        
        # Horizontal
        a = self.horizontal_params["a"]
        b = self.horizontal_params["b"]
        x_c = self.horizontal_params["x_c"]
        y_c = self.horizontal_params["y_c"]
        E_c = self.horizontal_params["E_c"]
        N_c = self.horizontal_params["N_c"]

        # Vertical
        C = self.vertical_params["vertical_shift"]
        Sn = self.vertical_params["slope_north"]
        Se = self.vertical_params["slope_east"]
        Nc = self.vertical_params["centroid_north"]
        Ec = self.vertical_params["centroid_east"]

        # Pydantic validation
        valid_pts = [PointTransform(**row) for row in df.to_dict('records')]

        # Handle column names dynamically
        if "Easting_global" in df.columns:
            x = df["Easting_global"].values
            y = df["Northing_global"].values
        else:
            x = df["Easting"].values
            y = df["Northing"].values
            
        # Vertical Height Selection (Global First)
        if "EllipsoidalHeight" in df.columns:
             h_input = df["EllipsoidalHeight"].values
        elif "Elevation" in df.columns:
             # Fallback for purely local ops (uncommon in strict transformations)
             h_input = df["Elevation"].values
        else:
             h_input = np.zeros(len(df))

        # Apply 2D Sim
        E_trans = a * (x - x_c) - b * (y - y_c) + E_c
        N_trans = b * (x - x_c) + a * (y - y_c) + N_c
        
        # Extrapolation Warning using ConvexHull
        # We need at least 3 points, non-collinear, to form a hull.
        local_pts_list = self.horizontal_params.get("local_control_points")
        if local_pts_list is not None:
            local_pts = np.array(local_pts_list)
            max_out_dist, count_out = check_extrapolation(E_trans, N_trans, local_pts)
            if max_out_dist is not None:
                import warnings
                warnings.warn(f"Extrapolación detectada: {count_out} puntos caen fuera del polígono de control. Distancia máxima al borde: {max_out_dist:.3f}m")
        
        # Apply Vertical Adjustment
        # We need N_local and E_local for the plane. 
        # If we only have global input (transforming global to local), we use the transformed values as proxy for position
        # Or if we have local columns in input.
        
        # The plane Z_err = C + Sn*(N - Nc) + Se*(E - Ec)
        # Z_local_derived = Z_global - Z_err (if subtracting error) or Z_local + Z_adj = Z_global
        # Let's align with the training: Z_error = Z_global - Z_local
        # So Z_global = Z_local + Z_error
        # We want to output "Transformed" coords. Usually this means transforming GNSS (Global) -> Local Grid.
        # But wait, TBC Site Cal transforms GPS (Global) to Grid (Local).
        # My train code defined a,b,tE,tN such that Local = f(Global).
        # So E_trans IS the estimated Local Easting.
        # So we can use E_trans and N_trans for the inclined plane domain.
        
        dZ = C + Sn * (N_trans - Nc) + Se * (E_trans - Ec)
        
        # wait, input 'h' in transform might be global h?
        # If input is global list, we want to match Local.
        # In train: Z_error = H_global - H_local.
        # This implies H_global = H_local + Z_error => H_local = H_global - Z_error.
        # If the user passes global coords to transform(), they expect Local Grid coords out.
        # So H_out = H_in (Global) - dZ.
        
        # Let's verify standard: Site Cal applied to GPS point:
        # 1. Convert Lat/Lon to Transverse Mercator (if not already) -> Global Grid
        # 2. Apply Horizontal Adjust -> Local Grid N,E
        # 3. Apply Vertical Adjust -> Local Elev.
        
        # If Z_error was defined as Global - Local (positive means Global is higher)
        # Then Local = Global - Z_error. 
        # Yes.
        
        H_trans = h_input - dZ
        # Let's assume input to transform is "Source" (Global).
        
        return pd.DataFrame({
            "Point": df["Point"],
            "Easting": E_trans,
            "Northing": N_trans,
            "h": H_trans
        })

    def transform_inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.horizontal_params is None or self.vertical_params is None:
            raise RuntimeError("The calibration model has not been trained.")
        
        # Horizontal
        a = self.horizontal_params["a"]
        b = self.horizontal_params["b"]
        x_c = self.horizontal_params["x_c"]
        y_c = self.horizontal_params["y_c"]
        E_c = self.horizontal_params["E_c"]
        N_c = self.horizontal_params["N_c"]

        # Vertical
        C = self.vertical_params["vertical_shift"]
        Sn = self.vertical_params["slope_north"]
        Se = self.vertical_params["slope_east"]
        Nc = self.vertical_params["centroid_north"]
        Ec = self.vertical_params["centroid_east"]

        # Handle column names dynamically
        if "Easting_local" in df.columns:
            E_in = df["Easting_local"].values
            N_in = df["Northing_local"].values
        else:
            E_in = df.get("Easting", np.zeros(len(df))).values
            N_in = df.get("Northing", np.zeros(len(df))).values
            
        if "Elevation" in df.columns:
            h_local = df["Elevation"].values
        elif "h" in df.columns:
            h_local = df["h"].values
        else:
            h_local = np.zeros(len(df))

        scale_sq = a**2 + b**2
        
        # Apply Inverse 2D Sim
        x_global = x_c + (a * (E_in - E_c) + b * (N_in - N_c)) / scale_sq
        y_global = y_c + (-b * (E_in - E_c) + a * (N_in - N_c)) / scale_sq
        
        # Apply Inverse Vertical
        dZ = C + Sn * (N_in - Nc) + Se * (E_in - Ec)
        H_global = h_local + dZ
        
        return pd.DataFrame({
            "Point": df["Point"] if "Point" in df.columns else np.arange(len(df)),
            "Easting_global": x_global,
            "Northing_global": y_global,
            "EllipsoidalHeight": H_global
        })

    def save(self) -> str:
        if self.horizontal_params is None or self.vertical_params is None:
            raise RuntimeError("The calibration model has not been trained.")
            
        data = {
            "method": "similarity2d",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "horizontal": self.horizontal_params,
                "vertical": self.vertical_params
            }
        }
        return json.dumps(data, indent=4)

    @classmethod
    def load(cls, json_str: str) -> 'Similarity2D':
        data = json.loads(json_str)
        if data.get("method") != "similarity2d":
            raise ValueError(f"Invalid method in sitecal file: {data.get('method')}")
            
        params = data.get("parameters", {})
        instance = cls()
        instance.horizontal_params = params.get("horizontal")
        instance.vertical_params = params.get("vertical")
        
        if not instance.horizontal_params or not instance.vertical_params:
            raise ValueError("Missing parameters in the loaded model.")
            
        return instance



class CalibrationFactory:
    @staticmethod
    def create(method: str) -> Calibration:
        if method == "default" or method == "ltm":
            return Similarity2D()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
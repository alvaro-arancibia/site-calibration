from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Calibration(ABC):
    @abstractmethod
    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        merged_df = pd.merge(df_local, df_global, on="Point", suffixes=('_local', '_global'))
        n = len(merged_df)

        # --- Horizontal (2D Similarity) ---
        x = merged_df["Easting_global"].values
        y = merged_df["Northing_global"].values
        E = merged_df["Easting_local"].values
        N = merged_df["Northing_local"].values
        
        # Determine weights (WLS)
        sigma_col = "sigma_global" if "sigma_global" in merged_df.columns else "sigma"
        if sigma_col in merged_df.columns:
            sigmas = merged_df[sigma_col].values
            sigmas_safe = np.where(sigmas > 0, sigmas, 1.0)
            w_sqrt = 1.0 / sigmas_safe
        else:
            w_sqrt = np.ones(n)
            
        w_sq = w_sqrt ** 2
        sum_w = np.sum(w_sq)

        # Calculate weighted centroids
        x_c = np.sum(w_sq * x) / sum_w
        y_c = np.sum(w_sq * y) / sum_w
        E_c = np.sum(w_sq * E) / sum_w
        N_c = np.sum(w_sq * N) / sum_w
        
        # Center coordinates
        x_prime = x - x_c
        y_prime = y - y_c
        E_prime = E - E_c
        N_prime = N - N_c

        # Validations for Horizontal Adjustment
        if n < 2:
            raise ValueError("Se requieren al menos 2 puntos para el ajuste horizontal.")

        coords_matrix = np.column_stack((E_prime, N_prime))
        # Rank < 2 means all centered points lie on a 1D line (collinear)
        if np.linalg.matrix_rank(coords_matrix, tol=1e-5) < 2 and n >= 3:
            raise ValueError("Geometría deficiente")

        # Solve for a and b using centered coordinates
        A = np.zeros((2 * n, 2))
        A[:n, 0] = x_prime
        A[:n, 1] = -y_prime
        A[n:, 0] = y_prime
        A[n:, 1] = x_prime

        L = np.concatenate([E_prime, N_prime])

        w_sqrt_2n = np.concatenate([w_sqrt, w_sqrt])
        A_w = A * w_sqrt_2n[:, np.newaxis]
        L_w = L * w_sqrt_2n

        params_ab, _, _, _ = np.linalg.lstsq(A_w, L_w, rcond=None)
        a = params_ab[0]
        b = params_ab[1]
        
        # --- Baarda Data Snooping (Horizontal) ---
        dof_h = 2 * n - 4
        outlier_flags_h = np.zeros(n, dtype=bool)
        
        if dof_h > 0:
            # Reconstruct V (residuals) for horizontal
            V_h = A_w @ params_ab - L_w
            
            # Varianza a posteriori
            sigma0_sq_h = np.dot(V_h, V_h) / dof_h
            
            if sigma0_sq_h > 1e-12:  # Avoid division by zero if perfect fit
                # N = A_w^T * A_w
                N_mat_h = A_w.T @ A_w
                
                try:
                    N_inv_h = np.linalg.inv(N_mat_h)
                    # Q_vv = I - A_w * N^-1 * A_w^T  (assuming W is already in A_w)
                    # We only need the diagonal of Q_vv
                    # diag(A_w * N^-1 * A_w^T)
                    diag_Q_hat = np.sum((A_w @ N_inv_h) * A_w, axis=1)
                    diag_Q_vv = 1.0 - diag_Q_hat
                    
                    # W_i statistic
                    # V returns a 2n vector (Easting array, then Northing array)
                    # We combine them per point to flag the point as outlier
                    # w_i = v_i / (sigma0 * sqrt(q_vvi))
                    
                    # Prevent division by zero or negative sqrt for q_vv
                    diag_Q_vv_safe = np.maximum(diag_Q_vv, 1e-12)
                    w_stat = V_h / (np.sqrt(sigma0_sq_h) * np.sqrt(diag_Q_vv_safe))
                    
                    # Combine Easting and Northing statistics per point (max of absolute values)
                    w_stat_E = w_stat[:n]
                    w_stat_N = w_stat[n:]
                    max_w_stat = np.maximum(np.abs(w_stat_E), np.abs(w_stat_N))
                    
                    # Pope's Tau or critical value. We use Baarda's canonical 3.0 or 3.29
                    critical_value = 3.0
                    outlier_flags_h = max_w_stat > critical_value
                except np.linalg.LinAlgError:
                    pass
        
        self.horizontal_params = {
            "a": a, "b": b,
            "x_c": x_c, "y_c": y_c,
            "E_c": E_c, "N_c": N_c,
            "local_control_points": np.column_stack((E, N))
        }

        # --- Vertical (Inclined Plane) ---
        # Z_error = Z_global - Z_local
        # Model: Z_error = C + S_N * (N_local - N_c) + S_E * (E_local - E_c)
        
        # Get heights. Strict Schema.
        h_global = merged_df["EllipsoidalHeight"].values 
        h_local = merged_df["Elevation"].values
        
        Z_error = h_global - h_local
        
        bad_vertical_geometry = False
        bad_condition = False
        
        if n >= 3:
            # Planar fit
            # A matrix: [1, (N - N_c), (E - E_c)]
            # We use local coordinates for the domain of the inclination as per standard practice (or translated global)
            # Standard software usually applies inclination based on position. Let's use Local Centering.
            
            A_v = np.ones((n, 3))
            A_v[:, 1] = N_prime # Using N_prime (N_local - N_c) is a good approximation for centering
            A_v[:, 2] = E_prime
            
            A_v_w = A_v * w_sqrt[:, np.newaxis]
            Z_error_w = Z_error * w_sqrt
            
            # Solve for [C, S_N, S_E]
            v_params, _, rank, _ = np.linalg.lstsq(A_v_w, Z_error_w, rcond=None)
            
            if rank < 3:
                bad_vertical_geometry = True
                # Fallback to constant shift if geometry is deficient
                C = np.sum(w_sq * Z_error) / sum_w
                slope_n = 0.0
                slope_e = 0.0
            else:
                cond = np.linalg.cond(A_v)
                if cond > 1e10:
                    bad_condition = True
    
                C = v_params[0]
                slope_n = v_params[1]
                slope_e = v_params[2]
        else:
            # Constant shift only
            C = np.sum(w_sq * Z_error) / sum_w
            slope_n = 0.0
            slope_e = 0.0
            rank = 1
            A_v_w = np.ones((n, 1)) * w_sqrt[:, np.newaxis]
            v_params = np.array([C])
            Z_error_w = Z_error * w_sqrt
            
        # --- Baarda Data Snooping (Vertical) ---
        # dof = n - (3 for plane, 1 for constant shift)
        num_v_params = 3 if (n >= 3 and rank >= 3) else 1
        dof_v = n - num_v_params
        outlier_flags_v = np.zeros(n, dtype=bool)
        
        if dof_v > 0 and 'A_v_w' in locals():
            V_v = A_v_w @ v_params - Z_error_w
            sigma0_sq_v = np.dot(V_v, V_v) / dof_v
            
            if sigma0_sq_v > 1e-12:
                N_mat_v = A_v_w.T @ A_v_w
                try:
                    N_inv_v = np.linalg.inv(N_mat_v)
                    diag_Q_hat_v = np.sum((A_v_w @ N_inv_v) * A_v_w, axis=1)
                    diag_Q_vv_v = 1.0 - diag_Q_hat_v
                    
                    diag_Q_vv_v_safe = np.maximum(diag_Q_vv_v, 1e-12)
                    w_stat_v = V_v / (np.sqrt(sigma0_sq_v) * np.sqrt(diag_Q_vv_v_safe))
                    
                    critical_value = 3.0
                    outlier_flags_v = np.abs(w_stat_v) > critical_value
                except np.linalg.LinAlgError:
                    pass
            
        self.vertical_params = {
            "vertical_shift": C,
            "slope_north": slope_n,
            "slope_east": slope_e,
            "centroid_north": N_c,
            "centroid_east": E_c,
            "rank": rank,
            "bad_geometry": bad_vertical_geometry,
            "bad_condition": bad_condition
        }
        
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
        # dH = Transformed (Local Calc) - Expected Local (Elevation)
        # Transformed["h"] is the calculated Local Height.
        # So dH = Transformed["h"] - Elevation
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
        local_pts = self.horizontal_params.get("local_control_points")
        if local_pts is not None and len(local_pts) >= 3:
            try:
                from scipy.spatial import ConvexHull
                from scipy.spatial.qhull import QhullError
                
                try:
                    hull = ConvexHull(local_pts)
                    
                    # Hull equations: A * x + B * y + C <= 0
                    equations = hull.equations
                    
                    # For each transformed point, check distance to all hull facets
                    trans_pts = np.column_stack((E_trans, N_trans))
                    
                    # Compute signed distances: (N_points, N_facets)
                    # dists = trans_pts @ normal + offset
                    dists = np.dot(trans_pts, equations[:, :2].T) + equations[:, 2]
                    
                    # A point is outside if the maximum distance to any facet is > 0 (plus epsilon)
                    max_dists = np.max(dists, axis=1)
                    
                    # Threshold for floating point inaccuracies
                    epsilon = 1e-5
                    
                    # Find points that are outside
                    outside_mask = max_dists > epsilon
                    if np.any(outside_mask):
                        import warnings
                        max_out_dist = np.max(max_dists[outside_mask])
                        warnings.warn(f"Extrapolación detectada: {np.sum(outside_mask)} puntos caen fuera del polígono de control. Distancia máxima al borde: {max_out_dist:.3f}m")
                        
                except QhullError:
                    # Geometry is likely degenerate (e.g. collinear), silently skip
                    pass
            except ImportError:
                # scipy not available, silently skip
                pass
        
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



class CalibrationFactory:
    @staticmethod
    def create(method: str) -> Calibration:
        if method == "default" or method == "ltm":
            return Similarity2D()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
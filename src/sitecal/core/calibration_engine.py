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
        self.params = None
        self.residuals = None

    def train(self, df_local: pd.DataFrame, df_global: pd.DataFrame):
        """
        Calculates 2D similarity transformation parameters (a, b, tE, tN)
        using least squares adjustment with centered coordinates for numerical stability.
        """
        
        merged_df = pd.merge(df_local, df_global, on="Point", suffixes=('_local', '_global'))

        x = merged_df["Easting_global"].values
        y = merged_df["Northing_global"].values
        E = merged_df["Easting_local"].values
        N = merged_df["Northing_local"].values
        
        # Calculate centroids
        x_c = np.mean(x)
        y_c = np.mean(y)
        E_c = np.mean(E)
        N_c = np.mean(N)
        
        # Center coordinates
        x_prime = x - x_c
        y_prime = y - y_c
        E_prime = E - E_c
        N_prime = N - N_c

        # Solve for a and b using centered coordinates
        n = len(merged_df)
        A = np.zeros((2 * n, 2))
        A[:n, 0] = x_prime
        A[:n, 1] = -y_prime
        A[n:, 0] = y_prime
        A[n:, 1] = x_prime

        L = np.concatenate([E_prime, N_prime])

        params_ab, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
        a = params_ab[0]
        b = params_ab[1]
        
        # Calculate translations
        tE = E_c - a * x_c + b * y_c
        tN = N_c - b * x_c - a * y_c

        self.params = {"a": a, "b": b, "tE": tE, "tN": tN}
        
        # Calculate residuals
        transformed = self.transform(merged_df)
        self.residuals = pd.DataFrame({
            "Point": merged_df["Point"],
            "dE": transformed["Easting"] - merged_df["Easting_local"],
            "dN": transformed["Northing"] - merged_df["Northing_local"],
            # In this model, dH is considered 0 as it's a 2D transformation
            "dH": np.zeros(n) 
        })


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("The calibration model has not been trained.")
        
        a = self.params["a"]
        b = self.params["b"]
        tE = self.params["tE"]
        tN = self.params["tN"]

        x = df["Easting_global"].values
        y = df["Northing_global"].values

        E_trans = a * x - b * y + tE
        N_trans = b * x + a * y + tN

        return pd.DataFrame({
            "Point": df["Point"],
            "Easting": E_trans,
            "Northing": N_trans,
            "h": df["h_global"].values if "h_global" in df.columns else np.zeros(len(df))
        })


class Helmert7(Calibration):
    def __init__(self):
        self.params = None
        self.residuals = None
        self.R = None
        self.t = None
        self.s = None

    def train(self, df_local_geo: pd.DataFrame, df_global_geo: pd.DataFrame):
        """
        Calculates 7-parameter Helmert transformation using SVD (Procrustes analysis).
        This method is robust against large rotations.
        """
        merged_df = pd.merge(df_local_geo, df_global_geo, on="Point", suffixes=('_local', '_global'))

        src = merged_df[["X_global", "Y_global", "Z_global"]].values
        dst = merged_df[["X_local", "Y_local", "Z_local"]].values

        # 1. Center both point clouds
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid

        # 2. Compute covariance matrix
        H = src_centered.T @ dst_centered

        # 3. Perform SVD
        U, _, Vt = np.linalg.svd(H)

        # 4. Calculate rotation matrix
        R = Vt.T @ U.T

        # 5. Check for reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 6. Calculate scale factor
        var_src = np.var(src_centered, axis=0).sum()
        s = np.trace(np.diag(_) @ R) / var_src
        
        # 7. Calculate translation
        t = dst_centroid - s * R @ src_centroid
        
        self.R = R
        self.t = t
        self.s = s

        self.params = {
            "tx": t[0], "ty": t[1], "tz": t[2],
            "s": s,
            "R11": R[0, 0], "R12": R[0, 1], "R13": R[0, 2],
            "R21": R[1, 0], "R22": R[1, 1], "R23": R[1, 2],
            "R31": R[2, 0], "R32": R[2, 1], "R33": R[2, 2],
        }
        
        # Calculate residuals
        transformed = self.transform(merged_df)
        self.residuals = pd.DataFrame({
            "Point": merged_df["Point"],
            "dE": transformed["X"] - merged_df["X_local"],
            "dN": transformed["Y"] - merged_df["Y_local"],
            "dH": transformed["Z"] - merged_df["Z_local"]
        })

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.R is None or self.t is None or self.s is None:
            raise RuntimeError("The calibration model has not been trained.")

        src_coords = df[["X_global", "Y_global", "Z_global"]].values
        
        # Apply the transformation: s * R @ P + t
        transformed_coords = (self.s * self.R @ src_coords.T).T + self.t
        
        return pd.DataFrame({
            "Point": df["Point"],
            "X": transformed_coords[:, 0],
            "Y": transformed_coords[:, 1],
            "Z": transformed_coords[:, 2]
        })

class CalibrationFactory:
    @staticmethod
    def create(method: str) -> Calibration:
        if method == "tbc" or method == "ltm":
            return Similarity2D()
        elif method == "helmert":
            return Helmert7()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

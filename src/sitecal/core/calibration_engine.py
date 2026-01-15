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

        # Handle column names dynamically (merged vs standalone)
        if "Easting_global" in df.columns:
            x = df["Easting_global"].values
            y = df["Northing_global"].values
        else:
            x = df["Easting"].values
            y = df["Northing"].values

        E_trans = a * x - b * y + tE
        N_trans = b * x + a * y + tN
        
        # Handle height pass-through (try _global suffix first, then plain)
        if "h_global" in df.columns:
            h_vals = df["h_global"].values
        elif "h" in df.columns:
            h_vals = df["h"].values
        else:
            h_vals = np.zeros(len(df))

        return pd.DataFrame({
            "Point": df["Point"],
            "Easting": E_trans,
            "Northing": N_trans,
            "h": h_vals
        })



class CalibrationFactory:
    @staticmethod
    def create(method: str) -> Calibration:
        if method == "tbc" or method == "ltm":
            return Similarity2D()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError


class Projection(ABC):
    @abstractmethod
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class Default(Projection):
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the first point as the origin (0,0) and a scale factor of 1.
        This creates a local Transverse Mercator projection centered on the project.
        """
        if df.empty:
            raise ValueError("Cannot project empty DataFrame. Need at least one point to define origin.")

        # Use the first point as the projection origin
        lat_0 = df.iloc[0]["Latitude"]
        lon_0 = df.iloc[0]["Longitude"]
        
        # Define projection: TM, Origin at 1st point, Scale 1.0, FE 0, FN 0
        proj_string = (
            f"+proj=tmerc +lat_0={lat_0} +lon_0={lon_0} "
            f"+k=1.0 +x_0=0 +y_0=0 "
            f"+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        
        src_crs = CRS("EPSG:4326")  # WGS84 Geodetic
        dst_crs = CRS(proj_string)

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        try:
            # Transform all points to this local system
            easting, northing = transformer.transform(df["Longitude"].values, df["Latitude"].values)
            
            df_out = df.copy()
            df_out["Easting"] = easting
            df_out["Northing"] = northing
            return df_out
            
        except ProjError as e:
            raise RuntimeError(f"Default Projection failed: {e}")


class UTM(Projection):
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Projects geodetic coordinates (Lat, Lon) to UTM.
        The UTM zone is automatically determined from the mean longitude.
        """
        if df.empty:
             raise ValueError("Cannot project empty DataFrame")

        lon_mean = df["Longitude"].mean()
        # Simple UTM zone calculation
        utm_zone = int((lon_mean + 180) / 6) + 1
        
        src_crs = CRS("EPSG:4326")  # WGS84
        # Assuming southern hemisphere for Chile/South America focus, 
        # but technically should check Lat. keeping simple for MVP.
        is_south = df["Latitude"].mean() < 0
        epsg_code = 32700 + utm_zone if is_south else 32600 + utm_zone
        
        dst_crs = CRS(f"EPSG:{epsg_code}")

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        try:
            easting, northing = transformer.transform(df["Longitude"].values, df["Latitude"].values)
            df_out = df.copy()
            df_out["Easting"] = easting
            df_out["Northing"] = northing
            return df_out
        except ProjError as e:
            raise RuntimeError(f"UTM Projection failed: {e}")


class LTM(Projection):
    def __init__(self, central_meridian, latitude_of_origin, false_easting, false_northing, scale_factor):
        self.central_meridian = central_meridian
        self.latitude_of_origin = latitude_of_origin
        self.false_easting = false_easting
        self.false_northing = false_northing
        self.scale_factor = scale_factor

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Projects geodetic coordinates to a custom LTM projection.
        """
        if df.empty:
             raise ValueError("Cannot project empty DataFrame")

        proj_string = (
            f"+proj=tmerc +lat_0={self.latitude_of_origin} +lon_0={self.central_meridian} "
            f"+k={self.scale_factor} +x_0={self.false_easting} +y_0={self.false_northing} "
            f"+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        
        src_crs = CRS("EPSG:4326")  # WGS84
        dst_crs = CRS(proj_string)

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        
        try:
            easting, northing = transformer.transform(df["Longitude"].values, df["Latitude"].values)
            df_out = df.copy()
            df_out["Easting"] = easting
            df_out["Northing"] = northing
            return df_out
        except ProjError as e:
            raise RuntimeError(f"LTM Projection failed: {e}")


class EPSG(Projection):
    def __init__(self, epsg_code: int):
        self.epsg_code = int(epsg_code)

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Projects geodetic coordinates (Lat, Lon) using any EPSG code via PyProj.
        """
        if df.empty:
            raise ValueError("Cannot project empty DataFrame")

        src_crs = CRS("EPSG:4326")
        try:
            dst_crs = CRS(f"EPSG:{self.epsg_code}")
        except Exception as e:
            raise ValueError(f"Invalid EPSG code {self.epsg_code}: {e}")

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        try:
            easting, northing = transformer.transform(df["Longitude"].values, df["Latitude"].values)
            df_out = df.copy()
            df_out["Easting"] = easting
            df_out["Northing"] = northing
            return df_out
        except ProjError as e:
            raise RuntimeError(f"EPSG:{self.epsg_code} Projection failed: {e}")


class ProjectionFactory:
    @staticmethod
    def create(method: str, **kwargs) -> Projection:
        if method == "default":
            return Default()
        elif method == "utm":
            return UTM()
        elif method == "ltm":
            return LTM(
                central_meridian=kwargs.get("central_meridian"),
                latitude_of_origin=kwargs.get("latitude_of_origin"),
                false_easting=kwargs.get("false_easting"),
                false_northing=kwargs.get("false_northing"),
                scale_factor=kwargs.get("scale_factor"),
            )
        elif method == "epsg":
            return EPSG(epsg_code=kwargs.get("epsg_code"))
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
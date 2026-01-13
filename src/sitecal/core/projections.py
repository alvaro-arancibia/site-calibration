from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError


class Projection(ABC):
    @abstractmethod
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class TBCDefault(Projection):
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the first point as the origin (0,0) and a scale factor of 1.
        This is the default behavior of Trimble Business Center.
        """
        # It is assumed that the local coordinates are already in this projection
        return df.copy()


class UTM(Projection):
    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Projects geodetic coordinates (Lat, Lon) to UTM.
        The UTM zone is automatically determined from the mean longitude.
        """
        lon_mean = df["Lon"].mean()
        utm_zone = int((lon_mean + 180) / 6) + 1
        
        src_crs = CRS("EPSG:4326")  # WGS84
        # Assuming southern hemisphere, which is correct for Chile
        dst_crs = CRS(f"EPSG:327{utm_zone}")

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        try:
            easting, northing = transformer.transform(df["Lon"].values, df["Lat"].values)
            df["Easting"] = easting
            df["Northing"] = northing
            return df
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
        proj_string = (
            f"+proj=tmerc +lat_0={self.latitude_of_origin} +lon_0={self.central_meridian} "
            f"+k={self.scale_factor} +x_0={self.false_easting} +y_0={self.false_northing} "
            f"+ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        
        src_crs = CRS("EPSG:4326")  # WGS84
        dst_crs = CRS(proj_string)

        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        
        try:
            easting, northing = transformer.transform(df["Lon"].values, df["Lat"].values)
            df["Easting"] = easting
            df["Northing"] = northing
            return df
        except ProjError as e:
            raise RuntimeError(f"LTM Projection failed: {e}")


class ProjectionFactory:
    @staticmethod
    def create(method: str, **kwargs) -> Projection:
        if method == "tbc":
            return TBCDefault()
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
        else:
            raise ValueError(f"Unknown projection method: {method}")

import pandas as pd
from pyproj import CRS, Transformer

def geodetic_to_geocentric(df: pd.DataFrame, epoch: float = 2020.0) -> pd.DataFrame:
    """
    Converts geodetic coordinates (Lat, Lon, h) to geocentric coordinates (X, Y, Z).

    Args:
        df: DataFrame with columns "Lat", "Lon", "h".
        epoch: The epoch of the coordinates. Currently not used in the transformation
               but included for future compatibility.

    Returns:
        A new DataFrame with geocentric coordinates "X", "Y", "Z".
    """
    # Source CRS: WGS 84 Geodetic
    # We are using WGS84, which is equivalent to ITRF at the meter level.
    # pyproj uses the EPSG codes to define the CRS.
    # WGS 84 geodetic CRS is EPSG:4979. The geocentric version is EPSG:4978
    src_crs = CRS("EPSG:4979")
    
    # Target CRS: WGS 84 Geocentric
    dst_crs = CRS("EPSG:4978")

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Note: pyproj transformers expect (lon, lat, h)
    x, y, z = transformer.transform(df["Lon"].values, df["Lat"].values, df["h"].values)

    result_df = pd.DataFrame({
        "Point": df["Point"],
        "X": x,
        "Y": y,
        "Z": z
    })
    
    return result_df

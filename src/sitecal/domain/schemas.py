from typing import Optional, List
from pydantic import BaseModel, Field, conlist, model_validator


class PointLocal(BaseModel):
    Point: str
    Easting: float = Field(alias="Easting_local")
    Northing: float = Field(alias="Northing_local")
    Elevation: float

    # Allow initializing without alias to match raw DataFrame
    model_config = {"populate_by_name": True}


class PointGlobal(BaseModel):
    Point: str
    Easting: float = Field(alias="Easting_global")
    Northing: float = Field(alias="Northing_global")
    EllipsoidalHeight: float
    sigma_global: Optional[float] = None
    sigma: Optional[float] = None

    model_config = {"populate_by_name": True}


class PointTransform(BaseModel):
    Point: str
    Easting_global: Optional[float] = None
    Northing_global: Optional[float] = None
    Easting: Optional[float] = None
    Northing: Optional[float] = None
    EllipsoidalHeight: Optional[float] = None
    Elevation: Optional[float] = None

    @model_validator(mode='after')
    def check_coordinates(self) -> 'PointTransform':
        has_global = self.Easting_global is not None and self.Northing_global is not None
        has_local = self.Easting is not None and self.Northing is not None
        if not (has_global or has_local):
            raise ValueError("Must provide either Easting_global/Northing_global or Easting/Northing")
        return self


class HorizontalParams(BaseModel):
    a: float
    b: float
    x_c: float
    y_c: float
    E_c: float
    N_c: float
    tE: float
    tN: float
    local_control_points: List[List[float]]


class VerticalParams(BaseModel):
    vertical_shift: float
    slope_north: float
    slope_east: float
    centroid_north: float
    centroid_east: float
    rank: int
    bad_geometry: bool
    bad_condition: bool

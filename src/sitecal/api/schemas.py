from pydantic import BaseModel, Field
import typing

class CalibrationParameters(BaseModel):
    a: float
    b: float
    tE: float
    tN: float

class ResidualPoint(BaseModel):
    Point: str
    dE: float
    dN: float
    dH: float

class CalibrationResult(BaseModel):
    parameters: CalibrationParameters
    residuals: typing.List[ResidualPoint]
    report: str

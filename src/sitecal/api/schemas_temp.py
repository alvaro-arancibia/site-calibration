import typing

class CalibrationParameters(typing.BaseModel if hasattr(typing, 'BaseModel') else object): # This is wrong, BaseModel is from pydantic
    a: float
    b: float
    tE: float
    tN: float

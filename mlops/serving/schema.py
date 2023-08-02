import pandera as pa
from pandera.typing import DataFrame, Index, Series
from pydantic import BaseModel


class InputSchema(pa.DataFrameModel):
    VendorID: Series[int] = pa.Field(coerce=True)
    passenger_count: Series[float] = pa.Field(coerce=True)
    trip_distance: Series[float] = pa.Field(coerce=True)
    RatecodeID: Series[float] = pa.Field(coerce=True)
    PULocationID: Series[int] = pa.Field(coerce=True)
    DOLocationID: Series[int] = pa.Field(coerce=True)
    payment_type: Series[int] = pa.Field(coerce=True)
    fare_amount: Series[float] = pa.Field(coerce=True)
    extra: Series[float] = pa.Field(coerce=True)
    mta_tax: Series[float] = pa.Field(coerce=True)
    tolls_amount: Series[float] = pa.Field(coerce=True)
    improvement_surcharge: Series[float] = pa.Field(coerce=True)
    total_amount: Series[float] = pa.Field(coerce=True)
    congestion_surcharge: Series[float] = pa.Field(coerce=True)
    Airport_fee: Series[float] = pa.Field(coerce=True)
    trip_time: Series[float] = pa.Field(coerce=True)
    ohe_Y: Series[bool] = pa.Field(coerce=True)


@pa.check_types
def transform(df: DataFrame[InputSchema]):
    return df.assign(revenue=100.0)


class PayloadSchema(BaseModel):
    VendorID: int
    passenger_count: float
    trip_distance: float
    RatecodeID: float
    PULocationID: int
    DOLocationID: int
    payment_type: int
    fare_amount: float
    extra: float
    mta_tax: float
    tolls_amount: float
    improvement_surcharge: float
    total_amount: float
    congestion_surcharge: float
    Airport_fee: float
    trip_time: float
    ohe_Y: bool

from datetime import datetime
from functools import wraps
from http import HTTPStatus

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder

from config.config import logger
from serving.schema import PayloadSchema, transform
from tagiftip import main

app = FastAPI(
    title="TIP or NO TIP",
    description="Classify wether the driver will receive a tip",
    version="0.0.1",
)


def construct_response(f):
    """Pretty response decorator"""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/")
@construct_response
def _index(request: Request) -> dict:
    """Health check"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.on_event("startup")
def _model() -> None:
    global model
    model = main.load_model(name="xgb_tip_no_tip", version=1)

    logger.info("Ready for inference!")


@app.post("/predict", tags=["Prediction"])
async def _predict(request: Request, payload: PayloadSchema) -> dict:
    """Model inference

    Args:
        request (Request): request
        payload (PayloadSchema): json request

    Returns:
        dict: status and prediction
    """
    payload_dict = {k: [v] for (k, v) in jsonable_encoder(payload).items()}
    df = pd.DataFrame.from_dict(payload_dict)

    df["VendorID"] = df["VendorID"].astype(
        "int32"
    )  # mlflow expects these 3 to be int32 else exception error
    df["PULocationID"] = df["PULocationID"].astype("int32")
    df["DOLocationID"] = df["PULocationID"].astype("int32")
    transform(df)  # double checks types using pandera

    pred = main.predict(model=model, data=df)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "predictions": pred.item(),
    }
    return response

from tagiftip import main
from fastapi import FastAPI, Request
from http import HTTPStatus
from datetime import datetime
from functools import wraps
from config.config import logger
from config import config

app = FastAPI()

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
def index(request : Request) -> dict:
    """Health check"""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.on_event("startup")
def load_artifacts():
    main.load_model(name="xgb_tip_no_tip", version=1)
    logger.info("Ready for inference!")
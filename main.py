from fastapi import FastAPI
# import numpy as np
from src.schema.predict import PredictRequest, PredictResponse

app = FastAPI()


FEET_TO_INCHES = 1728


@app.post(
    '/predict',
    response_model=PredictResponse
)
async def predict(
    body: PredictRequest
):
    width=40
    area = FEET_TO_INCHES*body.volume/width
    length = area**(0.5)
    height = length
    forecast_volume =height*length*width
    error = 100*abs(body.volume - forecast_volume)/body.volume
    return PredictResponse(
        length=length,
        width=width,
        height=height,
        n_boxes=1,
        error=error
    )
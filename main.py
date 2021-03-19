from fastapi import FastAPI
import numpy as np
from src.schema.predict import PredictRequest, PredictResponse

app = FastAPI()





@app.post(
    '/predict',
    response_model=PredictResponse
)
async def predict(
    body: PredictRequest
):
    width=40
    area = body.volume/width
    length = area**(0.5)
    height = length
    return PredictResponse(
        length=length,
        width=width,
        height=height,
        n_boxes=1
    )
from fastapi import FastAPI
import numpy as np
from src.schema.predict import PredictRequest, PredictResponse
from functools import lru_cache
import tensorflow as tf
import pickle 


app = FastAPI()

@lru_cache()
def load_model():
    model =  tf.keras.models.load_model('model')
    with open('scaler.pkl', 'rb') as pkl_file:
        scaler = pickle.load(pkl_file)
    return model, scaler

model, scaler = load_model()
FEET_TO_INCHES = 1728


@app.post(
    '/predict',
    response_model=PredictResponse
)
async def predict(
    body: PredictRequest
):
    body_np = np.array([[body.volume, body.weight]])
    body_np = scaler.transform(body_np)
    predictions = model.predict(body_np)
    print(predictions)
    n_boxes = int(predictions[0][0])
    width = predictions[0][1]
    height= predictions[0][2]
    length = FEET_TO_INCHES*body.volume/(width*height)
    forecast_volume =height*length*width/FEET_TO_INCHES
    error = 100*abs(body.volume - forecast_volume)/body.volume
    return PredictResponse(
        length=length,
        width=width,
        height=height,
        n_boxes=n_boxes,
        error=error
    )
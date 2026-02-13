from fastapi import FastAPI
import numpy as np
from tensorflow import keras

app = FastAPI()

model = keras.models.load("model.h5")

@app.post("/predict")

def predict(input_data:list):
    output = model.predict(np.array(input_data))
    return {"output": output.tolist()}



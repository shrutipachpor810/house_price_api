from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model with joblib (this is important!)
model = joblib.load("xgboost_house_price_model.pkl")  # Make sure this file exists

# Request schema
class HouseInput(BaseModel):
    bhk: int
    type: int
    locality: int
    area: float
    region: int
    status: int
    age: int

@app.get("/")
def root():
    return {"message": "PropPulse House Price Prediction API running!"}

@app.post("/predict")
def predict_price(data: HouseInput):
    input_data = np.array([[data.bhk, data.type, data.locality, data.area,
                            data.region, data.status, data.age]])
    prediction = model.predict(input_data)
    return {"predicted_price": round(float(prediction[0]), 2)}

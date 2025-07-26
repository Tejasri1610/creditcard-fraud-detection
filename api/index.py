# api/index.py

from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_event
import os
import sys

# Ensure root directory is in the path so inference.py works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Fraud Detector 🚀")

class TransactionInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float

@app.get("/")
def root():
    return {"message": "🚀 Fraud Detection API is up!"}

@app.post("/predict")
def predict(input_data: TransactionInput):
    input_list = list(input_data.dict().values())
    prediction = predict_event(input_list)
    return {
        "prediction": prediction,
        "meaning": "Fraud" if prediction == 1 else "Not Fraud ✅"
    }

# For Vercel compatibility
handler = app

import numpy as np
import joblib

# Load scaler and model directly (zip not allowed on Vercel)
scaler = joblib.load("scaler.pkl")
model = joblib.load("voting_model.pkl")

def predict_event(input_array):
    X = np.array(input_array).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return int(prediction[0])

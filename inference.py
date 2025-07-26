import zipfile
import os
import pickle
import numpy as np
import joblib

# Step 1: Extract zip if not already extracted
if not os.path.exists("voting_model.pkl"):
    with zipfile.ZipFile("voting_model.zip", "r") as zip_ref:
        zip_ref.extractall()

# Step 2: Load your scaler and model
scaler = joblib.load("scaler.pkl")

with open("voting_model.pkl", "rb") as f:
    model = pickle.load(f)

# Step 3: Define prediction function
def predict_event(input_array):
    X = np.array(input_array).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return int(prediction[0])

import joblib
import numpy as np

# ✅ Load your scaler and trained Voting model
scaler = joblib.load("scaler.pkl")
model = joblib.load("voting_model.pkl")

def predict_event(input_array):
    """
    input_array: should be a list or np.array of shape (n_features,)
    Example: [V1, V2, ..., V28, Amount, Time]
    """
    # Reshape for a single sample
    X = np.array(input_array).reshape(1, -1)

    # Scale with your saved scaler
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)

    return int(prediction[0])

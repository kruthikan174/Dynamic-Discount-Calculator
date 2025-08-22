# backend/ml_discount_logic.py

import joblib
import numpy as np
import os

# Load model (only once)
model_path = os.path.join("backend", "discount_model.pkl")
model = joblib.load(model_path)

def predict_discount(row):
    try:
        days = row["Days_To_Expiry"]
        pressure = row["Stock_Pressure"]
        slow = 1 if row["Turnover_Label"].lower() == "slow" else 0

        features = np.array([[days, pressure, slow]])
        discount = model.predict(features)[0]
        return min(max(round(discount), 0), 70)  # clamp to [0–70%]
    except:
        return 0  # fallback

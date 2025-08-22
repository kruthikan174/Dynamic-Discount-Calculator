import pandas as pd
import joblib
import numpy as np
import os

# === Greedy Logic ===
def calculate_discount_score(row):
    expiry_score = max(0, (15 - row["Days_To_Expiry"]) / 15)
    turnover_score = 1 if str(row["Turnover_Label"]).lower() == "slow" else 0
    pressure_score = min(row["Stock_Pressure"] / 3, 1)

    score = 0.5 * expiry_score + 0.3 * turnover_score + 0.2 * pressure_score
    discount = min(round(score * 100), 70)
    return discount

# === ML Model Load ===
model_path = os.path.join("backend", "discount_model", "discount_model.pkl")
model = joblib.load(model_path)

def predict_discount(row):
    try:
        days = row["Days_To_Expiry"]
        pressure = row["Stock_Pressure"]
        slow = 1 if str(row["Turnover_Label"]).lower() == "slow" else 0

        features = np.array([[days, pressure, slow]])
        discount = model.predict(features)[0]
        return min(max(round(discount), 0), 70)
    except:
        return 0

# === Filtering + Discount Logic ===
def get_filtered_inventory(df, filter_type, mode="greedy"):
    """
    Filters inventory based on type and applies discount logic based on mode.
    Discounts are only applied to non-expired items.
    """

    # Select records based on filter
    if filter_type == "expired":
        df = df[df["Is_Expired"] == True]

        # For expired items, set discount to 0 and mark final price same as unit price
        df["Discount_Percent"] = 0
        df["Dynamic_Selling_Price"] = df["Unit_Price"]
        return df

    else:
        # For all other filters, exclude expired
        df = df[df["Is_Expired"] == False]

        # Apply filter logic
        if filter_type == "near_expiry":
            df = df[df["Days_To_Expiry"] <= 10]
        elif filter_type == "low_turnover":
            df = df[df["Turnover_Label"].str.lower() == "slow"]
        elif filter_type == "high_pressure":
            df = df[df["Stock_Pressure"] > 1]

        # Apply discount
        if mode == "ml":
            df["Discount_Percent"] = df.apply(predict_discount, axis=1)
        else:
            df["Discount_Percent"] = df.apply(calculate_discount_score, axis=1)

        # Calculate final selling price
        df["Dynamic_Selling_Price"] = df["Unit_Price"] * (1 - df["Discount_Percent"] / 100)

        return df


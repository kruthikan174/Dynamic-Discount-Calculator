import streamlit as st
import numpy as np
import joblib

# === Load ML Model ===
model = joblib.load("backend/discount_model/discount_model.pkl")

# === Page Config ===
st.set_page_config(page_title="Discount Predictor", layout="centered")
st.markdown("# 🎯 Interactive Discount Predictor")
st.caption("Predict the best discount using Greedy or ML model")

st.markdown("Use the sliders below to simulate product conditions and see the recommended discount using either the **Greedy** or **ML** model.")

# === Slider Inputs ===
unit_price = st.slider("💲 Unit Price", min_value=10.0, max_value=1000.0, step=1.0, value=100.0)
stock_quantity = st.slider("📦 Stock Quantity", min_value=1, max_value=500, step=1, value=50)
turnover_ratio = st.slider("🔁 Turnover Ratio", min_value=0.0, max_value=10.0, step=0.1, value=1.5)
stock_pressure = st.slider("📊 Stock Pressure", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
days_to_expiry = st.slider("📅 Days to Expiry", min_value=0, max_value=90, step=1, value=10)

# === Derived Values ===
turnover_slow = 1 if turnover_ratio < 1.0 else 0

# === Model Choice ===
model_choice = st.radio("Choose Model to Apply", ["Greedy", "ML"], horizontal=True)

# === Discount Logic ===
def greedy_discount(days, pressure, turnover_slow):
    expiry_score = max(0, (15 - days) / 15)
    turnover_score = turnover_slow
    pressure_score = min(pressure / 3, 1)
    score = 0.5 * expiry_score + 0.3 * turnover_score + 0.2 * pressure_score
    return min(round(score * 100), 70)

def ml_discount(days, pressure, turnover_slow):
    features = np.array([[days, pressure, turnover_slow]])
    prediction = model.predict(features)[0]
    return int(min(max(round(prediction), 0), 70))

# === Display Result ===
if st.button("🎯 Predict Discount"):
    if model_choice == "Greedy":
        discount = greedy_discount(days_to_expiry, stock_pressure, turnover_slow)
        st.success(f"✅ **Greedy Model Suggests:** {discount}% Discount")
    else:
        discount = ml_discount(days_to_expiry, stock_pressure, turnover_slow)
        st.success(f"✅ **ML Model Suggests:** {discount}% Discount")

    final_price = unit_price * (1 - discount / 100)
    st.info(f"💰 Final Price After Discount: ₹{final_price:.2f}")

    st.markdown("---")
    st.markdown("**Inputs Summary:**")
    st.json({
        "Days to Expiry": days_to_expiry,
        "Stock Pressure": stock_pressure,
        "Turnover Ratio": turnover_ratio,
        "Turnover Slow": turnover_slow,
        "Unit Price": unit_price,
        "Stock Quantity": stock_quantity
    })

    st.markdown("🧠 Experiment with values to simulate real scenarios and explore model behavior!")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === Streamlit Page Settings ===
st.set_page_config(page_title="Discount Comparison Dashboard", layout="wide")

# === Load Data and Model ===
df = pd.read_csv("backend/Final_Normalized_Inventory.csv")
model_path = os.path.join("backend", "discount_model", "discount_model.pkl")
model = joblib.load(model_path)

# Filter out expired products for meaningful discounting
df = df[df["Is_Expired"] == False].copy()

# === Define Discount Functions ===
def greedy_discount(row):
    expiry_score = max(0, (15 - row["Days_To_Expiry"]) / 15)
    turnover_score = 1 if str(row["Turnover_Label"]).lower() == "slow" else 0
    pressure_score = min(row["Stock_Pressure"] / 3, 1)
    score = 0.5 * expiry_score + 0.3 * turnover_score + 0.2 * pressure_score
    return min(round(score * 100), 70)

def ml_discount(row):
    days = row["Days_To_Expiry"]
    pressure = row["Stock_Pressure"]
    slow = 1 if str(row["Turnover_Label"]).lower() == "slow" else 0
    features = np.array([[days, pressure, slow]])
    try:
        return min(max(round(model.predict(features)[0]), 0), 70)
    except:
        return 0

# === Apply Discounts ===
df["Greedy_Discount"] = df.apply(greedy_discount, axis=1)
df["ML_Discount"] = df.apply(ml_discount, axis=1)
df["Greedy_Price"] = df["Unit_Price"] * (1 - df["Greedy_Discount"] / 100)
df["ML_Price"] = df["Unit_Price"] * (1 - df["ML_Discount"] / 100)

# === Heading ===
st.title("🤖 Greedy vs ML Discount Engine Comparison")

# === Section 1: Discount Distribution ===
st.header("📈 1. Discount Distribution: Greedy vs ML")
st.markdown("""
This density plot compares the distribution of discount percentages predicted by the **Greedy approach** and the **ML model**.

- The x-axis shows the discount values.
- The y-axis shows how frequently those values occur (density).
- The **blue curve** is Greedy, and **orange** is ML.

The ML model generally offers higher discounts (right-shifted curve).
""")

fig1, ax1 = plt.subplots()
sns.kdeplot(df["Greedy_Discount"], label="Greedy", fill=True, alpha=0.5, color="blue")
sns.kdeplot(df["ML_Discount"], label="ML", fill=True, alpha=0.5, color="orange")
ax1.set_xlabel("Discount (%)")
ax1.set_ylabel("Density")
ax1.set_title("Discount Distribution: Greedy vs ML")
ax1.legend()
st.pyplot(fig1)

# === Section 2: Margin vs Discount ===
st.header("📉 2. Discount vs Profit Margin Correlation")
st.markdown("""
This dual scatter plot visualizes how discount rates affect profit margins for both strategies.

- The left plot shows **Greedy logic**, and the right one shows **ML logic**.
- Profit margin is calculated using a cost price that is 60% of unit price.

Both show a **negative correlation** – as discount increases, margin decreases.
""")

df["Cost_Price"] = df["Unit_Price"] * 0.6
df["Greedy_Margin"] = (df["Greedy_Price"] - df["Cost_Price"]) / df["Cost_Price"]
df["ML_Margin"] = (df["ML_Price"] - df["Cost_Price"]) / df["Cost_Price"]

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(x="Greedy_Discount", y="Greedy_Margin", data=df, ax=ax2[0], color="blue")
ax2[0].set_title("Greedy: Discount vs Margin")
ax2[0].set_xlabel("Discount (%)")
ax2[0].set_ylabel("Profit Margin")

sns.scatterplot(x="ML_Discount", y="ML_Margin", data=df, ax=ax2[1], color="orange")
ax2[1].set_title("ML: Discount vs Margin")
ax2[1].set_xlabel("Discount (%)")
ax2[1].set_ylabel("Profit Margin")

st.pyplot(fig2)

# === Section 3: Feature Importance ===
st.header("📊 3. ML Feature Importance")
st.markdown("""
This chart shows which features have the **most influence** on ML discount predictions.

- `Days_To_Expiry` and `Stock_Pressure` dominate.
- `Turnover_Slow` has relatively lower importance.

These insights help you understand the decision logic of the ML model.
""")

booster = model.get_booster()
importance = booster.get_score(importance_type="weight")
importance_df = pd.DataFrame.from_dict(importance, orient="index", columns=["Importance"])
importance_df.sort_values(by="Importance", ascending=False, inplace=True)
st.bar_chart(importance_df)

# === Section 4: Greedy Formula Explanation ===
st.header("📐 4. Greedy Discount Logic")
st.markdown("""
### 🧠 Greedy Logic Formula (Used for Comparison)

**Discount Score = 0.5 × Expiry_Score + 0.3 × Turnover_Score + 0.2 × Pressure_Score**

Where:  
- **Expiry_Score** = max(0, (15 - Days_To_Expiry) / 15)  
- **Turnover_Score** = 1 if Turnover_Label == "slow", else 0  
- **Pressure_Score** = min(Stock_Pressure / 3, 1)

This is a rule-based heuristic, offering explainability and control.
""")

import streamlit as st
import pandas as pd
import os
import sys

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.inventory_logic import get_filtered_inventory

# Load Data
df = pd.read_csv("backend/Final_Normalized_Inventory.csv")
df = df[df["Is_Expired"] == False]

st.title("📦 Active Inventory - Discount Engine")

# User selections
filter_type = st.selectbox("Choose Filter", [
    "near_expiry", "low_turnover", "high_pressure"
])
mode = st.selectbox("Discount Mode", ["greedy", "ml"])

# Apply discount logic and filter
filtered_df = get_filtered_inventory(df, filter_type, mode)

# ✅ Select relevant columns
display_cols = [
    "Product_Name", "Days_To_Expiry", "Turnover_Label", "Stock_Pressure",
    "Unit_Price", "Discount_Percent", "Dynamic_Selling_Price"
]

# ✅ Sort by highest discount first
filtered_df = filtered_df.sort_values(by="Discount_Percent", ascending=False)

# Show the cleaned and sorted dataframe
st.dataframe(filtered_df[display_cols])

st.markdown(f"**{len(filtered_df)} items shown with `{mode}` discount logic.**")

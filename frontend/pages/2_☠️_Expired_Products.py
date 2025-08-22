import streamlit as st
import pandas as pd

# Load expired data only
df = pd.read_csv("backend/Final_Normalized_Inventory.csv")
df = df[df["Is_Expired"] == True]

# ✅ Sort by Days_To_Expiry (ascending: most recent expiry first)
df = df.sort_values(by="Days_To_Expiry", ascending=False)

# ✅ Select relevant columns to display
display_cols = [
    "Product_Name", "Expiration_Date", "Days_To_Expiry", "Stock_Quantity", "Unit_Price"
]

# Display
st.title("☠️ Expired Products (Loss View)")
st.dataframe(df[display_cols])

st.markdown(f"**{len(df)} products are expired and unsellable.**")

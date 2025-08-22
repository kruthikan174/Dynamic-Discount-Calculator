## Dynamic Discount Calculator 

### 📌 Overview
This project implements a **Dynamic Discount Calculator** that predicts the optimal discount percentage for perishable goods based on expiry, demand, and sales patterns.  
The model helps reduce waste and maximize revenue by recommending timely discounts.

---

### 🤖 Model Used: **XGBoost Regressor**

#### Why XGBoost?
- Handles **tabular data** with both categorical + numerical features (e.g., expiry days, stock level, sales velocity).
- Performs well on **imbalanced datasets** (few items close to expiry vs many fresh ones).
- Includes **regularization techniques** to prevent overfitting — crucial since the dataset is relatively small.
- Provides **feature importance**, allowing clear explanation of discount drivers.

---

### ⚙️ How It Works
1. **Input Features**  
   - Days to expiry  
   - Current stock  
   - Historical sales velocity  
   - Demand trends  
   - Storage conditions  

2. **Target Variable**  
   - Optimal discount percentage (continuous regression output).  

3. **Training**  
   - Model trained on historical sales and expiry data.  
   - Objective: minimize **Mean Absolute Error (MAE)**.  

4. **Output Example**  
   - For a batch of mangoes with 2 days to expiry and low sales → model predicts `~15% discount`.  

---

### 🧠 Key Concepts
- **Gradient Boosting** → Builds multiple weak decision trees sequentially, each improving on the errors of the previous one.  
- **Feature Importance** → Explains which factors matter most (e.g., Expiry Days > Stock > Demand).  
- **Evaluation Metric** → Achieved **MAE = 10.1**, meaning predictions were on average off by ~10% discount margin.  

---

### 🚀 Applications
- Smart inventory management for retailers.  
- Waste reduction in supermarkets and e-commerce platforms.  
- Dynamic pricing in food & grocery supply chains.  

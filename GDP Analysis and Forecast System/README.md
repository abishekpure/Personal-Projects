# GDP Trend Analysis and Predictive Modeling System

An application designed to analyze historical GDP data, visualize long-term economic trends and generate 10-year forecasts using ARIMA time-series modeling. The system integrates directly with WBA to ensure access to reliable and consistent global economic data.

---

## Understanding GDP

**Gross Domestic Product (GDP)** represents the total monetary value of all goods and services produced within a country over a specific time period.  
It is widely used to:

- Measure economic performance  
- Compare countries' economic strength  
- Understand growth trends  
- Support evidence-based policy and decision-making  

Accurate GDP analysis helps identify long-term patterns, while forecasting assists governments, businesses, and researchers in planning for future economic scenarios.

---

## Workflow Overview: From Data Retrieval to Forecasting

### **1. Data Acquisition**
The application pulls official GDP data using the **World Bank API**, ensuring:
- Up-to-date economic indicators  
- Standardized structure across all countries  
- Reliable historical records dating back to 1960  

Users simply enter ISO country codes to load GDP datasets.

---

### **2. Data Cleaning & Preparation**
After retrieval, the system:
- Removes missing or null values  
- Organizes GDP values by year  
- Performs validation checks  
- Applies caching for faster repeated usage  

This ensures quality and consistency before visualization or modeling.

---

### **3. Visualization & Statistical Analysis**
Using **Plotly**, the dashboard presents:
- Global GDP line charts  
- Year filtering  
- Statistical evaluation of country/countries  

These visuals help users easily compare economic trajectories across regions and time.

---

### **4. ARIMA Model Selection**
To create accurate forecasts, the system uses an **automated ARIMA parameter search**.  
It scans through combinations of (p, d, q) values and selects the best configuration using **AIC minimization**, ensuring:

- Reduced overfitting  
- Improved predictive stability  
- No need for manual hyperparameter tuning  

---

### **5. Forecasting & Confidence Intervals**
Once the optimal ARIMA model is selected, the app generates:
- **10-year future GDP projections**  
- **Upper and lower confidence intervals**  
- A forecast plot overlaying all three curves  

This allows users to evaluate expected economic growth as well as the uncertainty around predictions.

---

## 🧠 Skills Used

- **Python**  
- **Streamlit**  
- **ARIMA Time-Series Forecasting**  
- **Plotly**  
- **API Integration**  

---

## ⚙️ Installation & Run

```bash
pip install -r requirements.txt
streamlit run app.py

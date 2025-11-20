import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)    # Always load NASA
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long


from statsmodels.tsa.arima.model import ARIMA

st.title("Model Comparison")

# Load dataset
df = st.session_state.get("df_long")
if df is None:
    st.error("Data not loaded. Please load data from Home page first.")
    st.stop()

# Prepare yearly averages
df_yearly = df.groupby("Year")["Temp_Anomaly"].mean().reset_index()
X = df_yearly["Year"].values.reshape(-1, 1)
y = df_yearly["Temp_Anomaly"].values

# Select prediction year
year_to_compare = st.slider("Select future year:", 2025, 2050, 2026)
future_X = np.array([[year_to_compare]])

# Train Models & Predict

# 1. Linear Regression
lin = LinearRegression().fit(X, y)
lin_pred = lin.predict(future_X)[0]

# 2. Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)
poly_pred = poly_model.predict(poly.transform(future_X))[0]

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=300).fit(X, y)
rf_pred = rf.predict(future_X)[0]

# 4. ARIMA
arima_model = ARIMA(y, order=(2, 1, 2)).fit()
arima_pred = arima_model.forecast()[0]

# Collect predictions
preds = {
    "Linear Regression": lin_pred,
    "Polynomial Regression": poly_pred,
    "Random Forest": rf_pred,
    "ARIMA": arima_pred
}

# Display predictions
st.subheader(f"Model Predictions for {year_to_compare}")
st.json(preds)

# -----------------------------
# Calculate Errors
# -----------------------------

errors = {}

# Linear Regression
errors["Linear Regression"] = {
    "RMSE": np.sqrt(mean_squared_error(y, lin.predict(X))),
    "MAE": mean_absolute_error(y, lin.predict(X)),
    "R2 Score": r2_score(y, lin.predict(X))
}

# Polynomial Regression
errors["Polynomial Regression"] = {
    "RMSE": np.sqrt(mean_squared_error(y, poly_model.predict(X_poly))),
    "MAE": mean_absolute_error(y, poly_model.predict(X_poly)),
    "R2 Score": r2_score(y, poly_model.predict(X_poly))
}

# Random Forest
errors["Random Forest"] = {
    "RMSE": np.sqrt(mean_squared_error(y, rf.predict(X))),
    "MAE": mean_absolute_error(y, rf.predict(X)),
    "R2 Score": r2_score(y, rf.predict(X))
}

# ARIMA (align fitted values)
arima_fitted = arima_model.fittedvalues
arima_true = y[-len(arima_fitted):]

errors["ARIMA"] = {
    "RMSE": np.sqrt(mean_squared_error(arima_true, arima_fitted)),
    "MAE": mean_absolute_error(arima_true, arima_fitted),
    "R2 Score": r2_score(arima_true, arima_fitted)
}

# Convert to DataFrame
err_df = pd.DataFrame(errors).T

# Show error table
st.subheader("Model Error Metrics")
st.dataframe(err_df)

# RMSE Bar Chart

st.subheader("RMSE Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(err_df.index, err_df["RMSE"], color=["pink", "seagreen", "orange", "brown"])
ax.set_ylabel("RMSE (Lower = Better)")
ax.set_title("RMSE Comparison of ML Models")
plt.xticks(rotation=15)

st.pyplot(fig)

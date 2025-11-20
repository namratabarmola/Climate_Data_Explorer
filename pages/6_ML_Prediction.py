# pages/6_ML_Prediction.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

from utils.analysis import yearly_mean, arima_forecast
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)    # Always load NASA
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long



st.title("ML Prediction (All Models Combined)")

# Load processed data
df = st.session_state.get("df_long")

if df is None:
    st.error("Data not loaded. Please go to the Home page first.")
else:
    # Prepare yearly mean data
    yearly = yearly_mean(df)
    X = yearly["Year"].values.reshape(-1, 1)
    y = yearly["Temp_Anomaly"].values

    # Select year to predict
    future = st.number_input(
        "Predict Temperature Anomaly for Year:",
        min_value=int(yearly.Year.max()) + 1,
        value=int(yearly.Year.max()) + 5
    )
    steps = future - int(yearly.Year.max())

    st.subheader("Model Predictions")

    # -----------------------------
    # 1) Linear Regression
    # -----------------------------
    lin_model = LinearRegression().fit(X, y)
    pred_lin = lin_model.predict([[future]])[0]

    # -----------------------------
    # 2) Polynomial Regression
    # -----------------------------
    degree = st.slider("Polynomial Degree", min_value=2, max_value=6, value=3)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression().fit(X_poly, y)
    pred_poly = poly_model.predict(poly.transform([[future]]))[0]

    # -----------------------------
    # 3) Random Forest Regression
    # -----------------------------
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X, y)
    pred_rf = rf_model.predict([[future]])[0]

    # -----------------------------
    # 4) ARIMA Forecast
    # -----------------------------
    series = yearly["Temp_Anomaly"]
    pred_arima = arima_forecast(series, steps=steps, order=(1, 1, 1)).iloc[-1]

    # -----------------------------
    # Show Predictions
    # -----------------------------
    st.write("Prediction Results")
    st.write(f"Linear Regression: {pred_lin:.4f} °C")
    st.write(f"Polynomial Regression (degree {degree}): {pred_poly:.4f} °C")
    st.write(f"Random Forest: {pred_rf:.4f} °C")
    st.write(f"ARIMA: {pred_arima:.4f} °C")

    # -----------------------------
    # Comparison Bar Chart
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Linear", "Polynomial", "Random Forest", "ARIMA"]
    values = [pred_lin, pred_poly, pred_rf, pred_arima]

    ax.bar(labels, values, color=["blue", "green", "orange", "red"])
    ax.set_title(f"Comparison of ML Models for Year {future}")
    ax.set_ylabel("Predicted Temperature Anomaly (°C)")
    st.pyplot(fig)

    #st.info("""
    #**Model Descriptions**
    #- **Linear Regression: Simple straight-line trend model.
    #- **Polynomial Regression**: Captures curved temperature trends.
    #- **Random Forest**: Non-linear model using decision trees.
    #- **ARIMA**: Classical time-series forecasting model.
    #""")

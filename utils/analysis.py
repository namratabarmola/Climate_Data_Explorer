# utils/analysis.py

import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.arima.model import ARIMA


# ------------------------------
# 1) YEARLY MEAN TEMPERATURE
# ------------------------------
def yearly_mean(df_long):
    """Returns yearly average temperature anomaly."""
    return df_long.groupby("Year", as_index=False)["Temp_Anomaly"].mean()


# ------------------------------
# 2) MONTHLY MEAN TEMPERATURE
# ------------------------------
def monthly_mean(df_long):
    """Returns average anomaly for each month across all years."""
    return df_long.groupby("Month", as_index=False)["Temp_Anomaly"].mean()


# ------------------------------
# 3) LINEAR TREND ANALYSIS
# ------------------------------
def compute_trend(yearly):
    """Applies simple linear regression for trend analysis."""
    x = yearly["Year"].astype(float)
    y = yearly["Temp_Anomaly"].astype(float)

    slope, intercept, r, p, _ = linregress(x, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r**2,
        "p": p
    }


# ------------------------------
# 4) OUTLIER DETECTION (IQR)
# ------------------------------
def detect_outliers(yearly):
    """Detects outlier years using the IQR (Interquartile Range) method."""
    q1 = yearly["Temp_Anomaly"].quantile(0.25)
    q3 = yearly["Temp_Anomaly"].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    return yearly[(yearly["Temp_Anomaly"] < low) | (yearly["Temp_Anomaly"] > high)]


# ------------------------------
# 5) ROLLING AVERAGE (5-year default)
# ------------------------------
def add_rolling_average(yearly, window=5):
    """Adds a rolling average column to yearly data."""
    df = yearly.copy()
    df["Rolling"] = df["Temp_Anomaly"].rolling(window=window, center=True, min_periods=1).mean()
    return df


# ------------------------------
# 6) ARIMA FORECASTING
# ------------------------------
def arima_forecast(series, steps=10, order=(1,1,1)):
    """
    Perform ARIMA forecasting.

    series : pandas Series of yearly temperature anomalies
    steps  : number of years to predict
    order  : ARIMA parameters (p,d,q)
    """
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        return pd.Series([None] * steps, name="ARIMA_Forecast")

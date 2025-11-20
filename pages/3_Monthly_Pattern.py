import streamlit as st
from utils.analysis import monthly_mean
from utils.visualization import plot_monthly
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)    # Always load NASA
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long

st.title("Monthly Temperature Pattern")

# Retrieve data from session_state
df = st.session_state.get("df_long")

if df is None:
    st.error("Data not loaded. Please go to the Home page first.")
else:
    st.subheader("Average Temperature Anomaly by Month")

    # Compute monthly averages
    monthly = monthly_mean(df)

    # Plot monthly trend
    st.pyplot(plot_monthly(monthly))

    st.subheader("Monthly Average Values")
    st.table(monthly)

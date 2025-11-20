import streamlit as st
from utils.analysis import yearly_mean, detect_outliers
from utils.visualization import plot_global_trend
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)    # Always load NASA
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long


st.title("Outliers")

df = st.session_state.get("df_long")
if df is None:
    st.error("Data not loaded.")
else:
    yearly = yearly_mean(df)
    out = detect_outliers(yearly)

    st.subheader("Outlier Years")
    st.table(out)

import streamlit as st
from utils.visualization import plot_heatmap
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)    # Always load NASA
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long


st.title("Heatmap")

df = st.session_state.get("df_long")
if df is None:
    st.error("Data not loaded.")
else:
    st.pyplot(plot_heatmap(df))

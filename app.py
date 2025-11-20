import streamlit as st
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

st.set_page_config(page_title="Climate Data Explorer", layout="wide")

st.title("Climate Data Explorer:An Interactive Analysis and Visualization Tool")

# --- Always Load NASA Dataset Automatically ---
df_raw = load_data(use_url=True)

# --- Process and Save to Session State ---
df_long = clean_and_convert(df_raw)
st.session_state.df_long = df_long

# Optional: Show small data preview
st.dataframe(df_long.head())

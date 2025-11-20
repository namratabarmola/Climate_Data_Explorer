import streamlit as st

st.title("About")
st.write("""
This project uses NASA global temperature anomaly data.
It demonstrates:
- Cleaning raw data
- Long-format conversion
- Trend analysis using linear regression
- Outlier detection using IQR
- Monthly correlation analysis
- Simple ML prediction
""")

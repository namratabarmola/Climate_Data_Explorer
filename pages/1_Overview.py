import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data
from utils.preprocess import clean_and_convert

if "df_long" not in st.session_state:
    df_raw = load_data(use_url=True)
    st.session_state.df_long = clean_and_convert(df_raw)

df = st.session_state.df_long


from utils.analysis import yearly_mean, add_rolling_average

st.title("Climate Data Dashboard – Overview")

df = st.session_state.get("df_long")

if df is None:
    st.error("Data not loaded.")
else:
    yearly = yearly_mean(df)
    yearly_roll = add_rolling_average(yearly)

    st.header("Climate Trend Summary")

    fig1, axes = plt.subplots(1, 2, figsize=(16, 5))


    #  --- Subplot A: Line plot

    axes[0].plot(yearly["Year"], yearly["Temp_Anomaly"], label="Yearly Data", color="blue")
    axes[0].plot(yearly_roll["Year"], yearly_roll["Rolling"], label="5-Year Rolling", color="red")
    axes[0].set_title("Temperature Trend")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Temp Anomaly (°C)")
    axes[0].legend()

    # --- Subplot B: Scatter Plot ---
    axes[1].scatter(yearly["Year"], yearly["Temp_Anomaly"], alpha=0.6, color="green")
    axes[1].set_title("Scatter Plot")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Temp Anomaly (°C)")

    st.pyplot(fig1)


    st.header("Temperature Distribution")

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

    # --- Subplot C: Histogram ---
    axes2[0, 0].hist(df["Temp_Anomaly"], bins=30, color="skyblue", edgecolor="black")
    axes2[0, 0].set_title("Histogram")

    # --- Subplot D: KDE Density Plot ---
    sns.kdeplot(df["Temp_Anomaly"], fill=True, ax=axes2[0, 1], color="orange")
    axes2[0, 1].set_title("Density (KDE)")

    # --- Subplot E: Boxplot By Month ---
    sns.boxplot(x=df["Month"], y=df["Temp_Anomaly"], ax=axes2[1, 0])
    axes2[1, 0].set_title("Boxplot by Month")

    # --- Subplot F: Monthly Heatmap ---
    heat = df.pivot_table(values="Temp_Anomaly", index="Year", columns="Month")
    sns.heatmap(heat, cmap="coolwarm", ax=axes2[1, 1])
    axes2[1, 1].set_title("Monthly Heatmap")

    st.pyplot(fig2)


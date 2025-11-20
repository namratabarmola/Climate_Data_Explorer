import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_global_trend(df_long):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df_long["Date"], df_long["Temp_Anomaly"], linewidth=0.9)
    ax.set_title("Global Temperature Anomaly Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.grid(True, alpha=0.3)
    return fig

def plot_monthly(monthly_df):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(monthly_df["Month"], monthly_df["Temp_Anomaly"], marker="o")
    ax.set_title("Average Temperature Anomaly by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temp Anomaly (°C)")
    ax.grid(alpha=0.3)
    return fig

def plot_heatmap(df_long):
    pivot = df_long.pivot_table(
        values="Temp_Anomaly", index="Year", columns="Month"
    )
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Monthly Temperature Anomalies")
    return fig

def plot_histogram(df_long):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df_long["Temp_Anomaly"], bins=30, color="orange", edgecolor="black")
    ax.set_title("Histogram of Temperature Anomalies")
    ax.set_xlabel("Temp Anomaly (°C)")
    ax.set_ylabel("Frequency")
    return fig


def plot_boxplot(df_long):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.boxplot(y=df_long["Temp_Anomaly"], color="skyblue")
    ax.set_title("Boxplot of Temperature Anomalies")
    ax.set_ylabel("Temp Anomaly (°C)")
    return fig


def plot_scatter(df_long):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df_long["Year"], df_long["Temp_Anomaly"], alpha=0.5)
    ax.set_title("Scatter Plot: Year vs Temperature Anomaly")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temp Anomaly (°C)")
    return fig


def plot_rolling_average(yearly):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(yearly["Year"], yearly["Temp_Anomaly"], label="Yearly Mean")
    ax.plot(yearly["Year"], yearly["Rolling"], linewidth=2, label="5-year Rolling Avg")
    ax.set_title("Rolling Average Temperature Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temp Anomaly (°C)")
    ax.legend()
    return fig

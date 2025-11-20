# utils/preprocess.py
import pandas as pd
import numpy as np

month_map = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
}

def clean_and_convert(df):
    df = df.copy()

    # Drop any duplicate year columns (like Year.1)
    df = df.loc[:, ~df.columns.str.contains('Year.1', case=False)]

    # Replace *** with NaN
    df.replace('***', np.nan, inplace=True)

    # Convert all numeric columns except 'Year'
    for col in df.columns:
        if col != "Year":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Year'])

    # Melt into long format
    df_long = df.melt(id_vars="Year", var_name="Month", value_name="Temp_Anomaly")

    # Filter valid months
    df_long = df_long[df_long["Month"].str[:3].isin(month_map.keys())]

    df_long["Month"] = df_long["Month"].str.slice(0, 3).map(month_map)

    # Build datetime
    df_long["Date"] = pd.to_datetime(
        dict(year=df_long["Year"], month=df_long["Month"], day=1)
    )

    df_long = df_long.sort_values("Date").reset_index(drop=True)
    return df_long

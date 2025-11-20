import pandas as pd
from pathlib import Path

NASA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

def load_data(use_url=True, local_path="data/global_temperature.csv"):
    if use_url:
        return pd.read_csv(NASA_URL, skiprows=1)
    else:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"Local file not found at {local_path}")
        return pd.read_csv(p)

# src/view_data.py
# Purpose: Load and display saved market data in the terminal

import pandas as pd
from pathlib import Path

# path to your saved dataset
DATA_FILE = Path("data/market.parquet")

# load the data
df = pd.read_parquet(DATA_FILE)

# print information to terminal
print("\n✅ Successfully loaded market data\n")
print("Shape:", df.shape)
print("\n--- HEAD (first 5 rows) ---")
print(df.head())
print("\n--- SUMMARY ---")
print(df.describe())

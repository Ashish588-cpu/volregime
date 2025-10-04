# ====================================
# test_data.py
# Quick check: load saved parquet file and preview contents
# ====================================

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def main():
    # Load parquet file
    df = pd.read_parquet(DATA_DIR / "market.parquet")

    # Show first 5 rows
    print("\n🔹 First rows:")
    print(df.head())

    # Show last 5 rows
    print("\n🔹 Last rows:")
    print(df.tail())

    # Summary info
    print("\n🔹 Data Summary:")
    print(df.info())

if __name__ == "__main__":
    main()

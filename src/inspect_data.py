# inspect_data.py
# check parquet file created by data_fetch.py
# this helps us confirm data is saved and structured properly

# pandas for data handling
# pathlib for file path handling
import pandas as pd
from pathlib import Path

# --- setup paths ---
# go up one level from /src folder to project root
ROOT = Path(__file__).resolve().parents[1]
# point to data/ folder
DATA_DIR = ROOT / "data"
# point to the parquet file created earlier
file_path = DATA_DIR / "market.parquet"

def main():
    # load parquet into pandas dataframe
    df = pd.read_parquet(file_path)

    # success message
    print("\n✅ Successfully loaded market.parquet\n")

    # print shape -> (rows, columns)
    print("Shape of dataset:", df.shape)

    # print first 5 rows
    print("\n--- Head (first 5 rows) ---")
    print(df.head())

    # print last 5 rows
    print("\n--- Tail (last 5 rows) ---")
    print(df.tail())

    # print all column names
    print("\n--- Columns ---")
    print(df.columns.tolist())

    # optional quick check: print summary stats
    print("\n--- Summary statistics ---")
    print(df.describe())

if __name__ == "__main__":
    main()

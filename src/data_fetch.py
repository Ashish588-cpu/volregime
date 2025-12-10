 # ================================
# data_fetch.py
# ================================
# Main data module — all other parts of the project depend on this.
#
# This script:
# 1) Downloads financial data (SPY and ^VIX) from Yahoo Finance
# 2) Extracts"Adjusted Close" prices (dividend & split adjusted)
# 3) Saves clean, aligned data to"data/market.parquet"
#
# Libraries:
# pathlib.Path → safely handle file paths across OS
# pandas → data manipulation and time series handling
# yfinance → fetches data from Yahoo Finance
# =================================

# ============================================================
# 1️⃣ Setup
# ============================================================
from pathlib import Path
import pandas as pd
import yfinance as yf

# Get project root folder (volregime/)
ROOT = Path(__file__).resolve().parents[1]

# Create a /data folder if it doesn’t already exist
DATA_DIR = ROOT /"data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2️⃣ Configuration
# ============================================================

# Tickers we’ll fetch — keeping it simple for now
# SPY → S&P 500 ETF (represents stock market)
# ^VIX → Volatility Index (market fear gauge)
TICKERS = ["SPY","^VIX"]

# Time range for dataset
START ="2015-01-01"
END = None # None means up to today


# ============================================================
# 3️⃣ Function to fetch Adjusted Close prices
# ============================================================

def fetch_adj_close(tickers, start=None, end=None):
"""
 Downloads Adjusted Close prices from Yahoo Finance.

 Args:
 tickers (list): e.g. ["SPY","^VIX"]
 start (str): start date in"YYYY-MM-DD"
 end (str): end date, or None for today

 Returns:
 DataFrame: trading dates × ticker columns
"""
 data = {}

 for t in tickers:
 try:
 # Download full dataset
 df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)

 # Check validity
 if isinstance(df, pd.DataFrame) and not df.empty and"Adj Close" in df.columns:
 data[t] = df["Adj Close"]
 print(f" Successfully fetched {t} ({len(df)} rows)")
 else:
 print(f"️ Skipping {t}: missing or invalid data")

 except Exception as e:
 print(f" Failed to fetch {t}: {e}")

 # Only combine valid tickers
 if not data:
 print(" No valid tickers fetched. Returning empty DataFrame.")
 return pd.DataFrame()

 # Combine into single DataFrame (aligns dates automatically)
 df = pd.concat(data, axis=1)

 # Reindex to business days (Mon–Fri only)
 if not df.empty:
 idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
 df = df.reindex(idx)

 return df


# ============================================================
# 4️⃣ Main execution block
# ============================================================

def main():
"""
 1. Fetch market data for SPY and ^VIX.
 2. Check that it’s valid.
 3. Save the clean dataset as a .parquet file.
"""

 # Step 1 → Fetch data
 df = fetch_adj_close(TICKERS, START, END)

 # Step 2 → Basic validation
 if df.empty or df.isna().all().all():
 print(" No valid data fetched. Skipping save.")
 return

 # Step 3 → Save output file
 out_file = DATA_DIR /"market.parquet"
 df.to_parquet(out_file)

 print("------------------------------------------------------------")
 print(f" Market data saved successfully to: {out_file}")
 print(f" Final dataset shape: {df.shape} (rows × columns)")
 print("------------------------------------------------------------")


# ============================================================
# 5️⃣ Entry point (runs only if called directly)
# ============================================================

if __name__ =="__main__":
 main()

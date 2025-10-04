# main data module
# every other module(features, labeling, modeling) depeonds on this

from pathlib import Path
import yfinance as yf
import pandas as pd

# --- Setup paths for data ---
ROOT = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # make sure "data/" exists

# --- Config ---
TICKERS = ["SPY", "^VIX", "TLT"]
START = "2015-01-01"
END = None   # None = up to today


def fetch_adj_close(tickers, start=None, end=None):
    """
    Download Adjusted Close prices from Yahoo Finance.
    Returns: DataFrame with tickers as columns.
    """
    df = yf.download(tickers, start=start, end=end)["Adj Close"]
    df = df.sort_index()
    return df


def main():
    # Step 1: fetch
    df = fetch_adj_close(TICKERS, START, END)

    # Step 2: save to parquet
    out_file = DATA_DIR / "market.parquet"
    df.to_parquet(out_file)

    print(f"✅ Saved data to {out_file}")


if __name__ == "__main__":
    main()

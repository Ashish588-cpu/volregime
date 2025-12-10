"""
Core data utilities for VolRegime financial platform
Production-ready data fetching with comprehensive error handling

This module provides:
- Robust API error handling with user-friendly messages
- Cached data fetching for performance
- Technical indicator calculations
- Data validation and fallback mechanisms
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import feedparser
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CACHE TIMESTAMP TRACKING
# ============================================================================

# Store cache timestamps for display in UI
_cache_timestamps = {}

def get_cache_timestamp(cache_key: str) -> Optional[datetime]:
"""Get the timestamp when a cache was last updated"""
 return _cache_timestamps.get(cache_key)

def set_cache_timestamp(cache_key: str) -> None:
"""Set the current timestamp for a cache key"""
 _cache_timestamps[cache_key] = datetime.now()

def get_last_data_update() -> Optional[datetime]:
"""Get the most recent cache update timestamp"""
 if not _cache_timestamps:
 return None
 return max(_cache_timestamps.values())


# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def handle_api_error(error_message: str, ticker: str = None) -> str:
"""
 Generate user-friendly HTML error message for API failures

 This function creates a styled error card that can be displayed
 in the Streamlit UI when data fetching fails.

 Args:
 error_message (str): Description of the error
 ticker (str, optional): Stock ticker symbol if applicable

 Returns:
 str: HTML string for rendering with st.markdown(unsafe_allow_html=True)

 Example:
 st.markdown(handle_api_error("Unable to fetch market data","AAPL"),
 unsafe_allow_html=True)
"""
 ticker_info = f" (Ticker: {ticker})" if ticker else""

 error_html = f"""
 <div style="background: rgba(220, 38, 38, 0.1); border: 1px solid #dc2626; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
 <p style="color: #dc2626; margin: 0; font-family: Poppins, sans-serif;">
 ️ <strong>Data Unavailable</strong><br>
 {error_message}{ticker_info}
 </p>
 </div>
"""
 return error_html


def log_error(error: Exception, context: str, ticker: str = None) -> None:
"""
 Log error details to console for debugging

 Args:
 error (Exception): The exception that was raised
 context (str): Description of what operation failed
 ticker (str, optional): Stock ticker symbol if applicable
"""
 ticker_info = f" [{ticker}]" if ticker else""
 logger.error(f"[VolRegime]{ticker_info} {context}: {type(error).__name__} - {str(error)}")


def safe_float(value, default: float = 0.0) -> float:
"""
 Safely convert value to float with fallback

 Args:
 value: Value to convert
 default (float): Default value if conversion fails

 Returns:
 float: Converted value or default
"""
 try:
 if value is None or (isinstance(value, float) and np.isnan(value)):
 return default
 return float(value)
 except (ValueError, TypeError):
 return default


# ============================================================================
# CORE DATA FETCHING WITH ROBUST ERROR HANDLING
# ============================================================================

@st.cache_data(ttl=600, show_spinner=False) # Cache for 10 minutes (historical data)
def get_stock_data(symbol: str, period: str ="1y", show_errors: bool = False) -> pd.DataFrame:
"""
 Fetch stock data with comprehensive error handling and validation

 Args:
 symbol (str): Stock ticker symbol
 period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
 show_errors (bool): Whether to display errors in UI (default: False for silent fails)

 Returns:
 pd.DataFrame: Stock price data with OHLCV columns, empty DataFrame on failure
"""
 try:
 # Validate symbol input
 if not symbol or not isinstance(symbol, str):
 log_error(ValueError("Invalid symbol"),"Symbol validation failed", symbol)
 return pd.DataFrame()

 # Normalize symbol
 symbol = symbol.upper().strip()

 # Validate period
 valid_periods = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
 if period.lower() not in valid_periods:
 log_error(ValueError(f"Invalid period: {period}"),"Period validation failed", symbol)
 period ="1y" # Default fallback

 # Fetch data from yfinance
 ticker = yf.Ticker(symbol)
 data = ticker.history(period=period)

 # Validate response data
 if data is None or data.empty:
 logger.warning(f"[VolRegime] No data returned for {symbol} (period: {period})")
 return pd.DataFrame()

 # Ensure required columns exist
 required_cols = ['Open','High','Low','Close','Volume']
 missing_cols = [col for col in required_cols if col not in data.columns]
 if missing_cols:
 log_error(ValueError(f"Missing columns: {missing_cols}"),"Data structure validation failed", symbol)
 return pd.DataFrame()

 # Clean data - remove NaN rows
 data = data.dropna(subset=['Open','High','Low','Close'])

 # Ensure datetime index
 if not isinstance(data.index, pd.DatetimeIndex):
 data.index = pd.to_datetime(data.index)

 # Track cache timestamp
 set_cache_timestamp(f"stock_data_{symbol}_{period}")
 logger.info(f"[VolRegime] Successfully fetched {len(data)} rows for {symbol}")
 return data

 except ConnectionError as e:
 log_error(e,"Network connection failed", symbol)
 if show_errors:
 st.error(f"Network error: Unable to reach market data servers for {symbol}")
 return pd.DataFrame()

 except TimeoutError as e:
 log_error(e,"Request timeout", symbol)
 if show_errors:
 st.warning(f"Request timeout for {symbol}. Please try again.")
 return pd.DataFrame()

 except Exception as e:
 log_error(e,"Unexpected error fetching stock data", symbol)
 if show_errors:
 st.error(f"Error fetching data for {symbol}: {str(e)}")
 return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False) # Cache for 1 minute
def get_current_price(symbol: str, show_errors: bool = False) -> Optional[float]:
"""
 Get current stock price with fallback mechanisms

 Args:
 symbol (str): Stock ticker symbol
 show_errors (bool): Whether to display errors in UI

 Returns:
 float: Current price or None if unavailable
"""
 try:
 if not symbol or not isinstance(symbol, str):
 return None

 symbol = symbol.upper().strip()
 ticker = yf.Ticker(symbol)

 # Try intraday data first (most current)
 try:
 data = ticker.history(period="1d", interval="1m")
 if data is not None and not data.empty and'Close' in data.columns:
 return safe_float(data['Close'].iloc[-1])
 except Exception:
 pass # Silently fall through to daily data

 # Fallback to daily data
 try:
 data = ticker.history(period="2d")
 if data is not None and not data.empty and'Close' in data.columns:
 return safe_float(data['Close'].iloc[-1])
 except Exception:
 pass

 logger.warning(f"[VolRegime] Could not retrieve current price for {symbol}")
 return None

 except Exception as e:
 log_error(e,"Error fetching current price", symbol)
 if show_errors:
 st.error(f"Error fetching current price for {symbol}")
 return None


@st.cache_data(ttl=1800, show_spinner=False) # Cache for 30 minutes (company info)
def get_stock_info(symbol: str, show_errors: bool = False) -> Dict:
"""
 Get comprehensive stock information with error handling

 Args:
 symbol (str): Stock ticker symbol
 show_errors (bool): Whether to display errors in UI

 Returns:
 dict: Stock information with fallbacks for missing data
"""
 # Default fallback data structure
 defaults = {
'longName': symbol if symbol else'Unknown',
'shortName': symbol if symbol else'Unknown',
'sector':'Unknown',
'industry':'Unknown',
'marketCap': 0,
'beta': 1.0,
'trailingPE': 0,
'forwardPE': 0,
'dividendYield': 0,
'fiftyTwoWeekHigh': 0,
'fiftyTwoWeekLow': 0,
'averageVolume': 0,
'previousClose': 0,
'volume': 0,
'longBusinessSummary': f'No description available for {symbol}',
'fullTimeEmployees': 0,
'website':''
 }

 try:
 if not symbol or not isinstance(symbol, str):
 return defaults

 symbol = symbol.upper().strip()
 ticker = yf.Ticker(symbol)
 info = ticker.info

 # Handle case where info is None or empty
 if not info:
 logger.warning(f"[VolRegime] No info returned for {symbol}")
 return defaults

 # Merge with defaults for missing fields
 for key, default_value in defaults.items():
 if key not in info or info[key] is None:
 info[key] = default_value

 # Track cache timestamp
 set_cache_timestamp(f"stock_info_{symbol}")
 logger.info(f"[VolRegime] Successfully fetched info for {symbol}")
 return info

 except Exception as e:
 log_error(e,"Error fetching stock info", symbol)
 if show_errors:
 st.error(f"Error fetching info for {symbol}")
 return defaults

@st.cache_data(ttl=300, show_spinner=False) # Cache for 5 minutes
def get_market_indices(show_errors: bool = False) -> Optional[Dict]:
"""
 Get major market indices data with comprehensive error handling

 Args:
 show_errors (bool): Whether to display errors in UI

 Returns:
 dict: Dictionary with market index data, or None if complete failure
"""
 indices = {
'S&P 500':'^GSPC',
'NASDAQ':'^IXIC',
'Dow Jones':'^DJI',
'VIX':'^VIX',
'Bitcoin':'BTC-USD',
'Ethereum':'ETH-USD'
 }

 market_data = {}
 successful_fetches = 0

 for name, symbol in indices.items():
 try:
 data = get_stock_data(symbol, period="5d", show_errors=False)

 if data is not None and not data.empty and len(data) >= 1:
 # Safely extract values with fallbacks
 current_price = safe_float(data['Close'].iloc[-1])
 prev_close = safe_float(data['Close'].iloc[-2]) if len(data) > 1 else current_price

 # Avoid division by zero
 if prev_close > 0:
 change = current_price - prev_close
 change_pct = (change / prev_close) * 100
 else:
 change = 0
 change_pct = 0

 volume = safe_float(data['Volume'].iloc[-1]) if'Volume' in data.columns else 0

 market_data[name] = {
'symbol': symbol,
'price': current_price,
'change': change,
'change_pct': change_pct,
'volume': volume,
'data': data
 }
 successful_fetches += 1
 else:
 # Provide fallback data structure
 market_data[name] = {
'symbol': symbol,
'price': 0,
'change': 0,
'change_pct': 0,
'volume': 0,
'data': pd.DataFrame()
 }
 logger.warning(f"[VolRegime] No data available for index: {name}")

 except Exception as e:
 log_error(e, f"Error fetching market index {name}", symbol)
 # Provide fallback data structure
 market_data[name] = {
'symbol': symbol,
'price': 0,
'change': 0,
'change_pct': 0,
'volume': 0,
'data': pd.DataFrame()
 }

 # Return None only if ALL fetches failed
 if successful_fetches == 0:
 logger.error("[VolRegime] Failed to fetch any market index data")
 if show_errors:
 st.error("Unable to fetch market data. Please check your internet connection.")
 return None

 # Track cache timestamp
 set_cache_timestamp("market_indices")
 logger.info(f"[VolRegime] Successfully fetched {successful_fetches}/{len(indices)} market indices")
 return market_data

# ============================================================================
# TECHNICAL INDICATORS WITH MATHEMATICAL PRECISION
# ============================================================================

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
"""
 Calculate comprehensive technical indicators with proper error handling

 Args:
 data (pd.DataFrame): Stock price data with OHLCV columns

 Returns:
 pd.DataFrame: Data with technical indicators added
"""
 if data.empty or len(data) < 50: # Need minimum data for indicators
 return data

 try:
 df = data.copy()

 # Moving Averages
 df['SMA_10'] = df['Close'].rolling(window=10, min_periods=10).mean()
 df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean()
 df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()
 df['SMA_200'] = df['Close'].rolling(window=200, min_periods=200).mean()

 # Exponential Moving Averages
 df['EMA_12'] = df['Close'].ewm(span=12, min_periods=12).mean()
 df['EMA_26'] = df['Close'].ewm(span=26, min_periods=26).mean()

 # RSI (Relative Strength Index)
 delta = df['Close'].diff()
 gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
 loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
 rs = gain / loss.replace(0, np.inf) # Avoid division by zero
 df['RSI'] = 100 - (100 / (1 + rs))

 # MACD (Moving Average Convergence Divergence)
 df['MACD'] = df['EMA_12'] - df['EMA_26']
 df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
 df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

 # Bollinger Bands
 df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=20).mean()
 bb_std = df['Close'].rolling(window=20, min_periods=20).std()
 df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
 df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
 df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
 df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']

 # Average True Range (ATR)
 high_low = df['High'] - df['Low']
 high_close = np.abs(df['High'] - df['Close'].shift())
 low_close = np.abs(df['Low'] - df['Close'].shift())
 ranges = pd.concat([high_low, high_close, low_close], axis=1)
 true_range = ranges.max(axis=1)
 df['ATR'] = true_range.rolling(window=14, min_periods=14).mean()

 # Volatility (20-day rolling)
 df['Volatility'] = df['Close'].pct_change().rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100

 # Volume indicators
 df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=20).mean()
 df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

 # Price momentum
 df['Price_Change_1D'] = df['Close'].pct_change(1) * 100
 df['Price_Change_5D'] = df['Close'].pct_change(5) * 100
 df['Price_Change_20D'] = df['Close'].pct_change(20) * 100

 return df

 except Exception as e:
 st.error(f"Error calculating technical indicators: {str(e)}")
 return data

def get_trend_classification(data: pd.DataFrame) -> str:
"""
 Classify trend based on moving averages and price action

 Args:
 data (pd.DataFrame): Stock data with technical indicators

 Returns:
 str: Trend classification (Uptrend, Downtrend, Sideways)
"""
 if data.empty or len(data) < 50:
 return"Insufficient Data"

 try:
 latest = data.iloc[-1]

 # Check if we have required indicators
 if'SMA_20' not in data.columns or'SMA_50' not in data.columns:
 return"Insufficient Data"

 current_price = latest['Close']
 sma_20 = latest['SMA_20']
 sma_50 = latest['SMA_50']

 # Skip if indicators are NaN
 if pd.isna(sma_20) or pd.isna(sma_50):
 return"Insufficient Data"

 # Trend classification logic
 if current_price > sma_20 > sma_50:
 return"Uptrend"
 elif current_price < sma_20 < sma_50:
 return"Downtrend"
 else:
 return"Sideways"

 except Exception as e:
 return"Error"

@st.cache_data(ttl=600, show_spinner=False) # Cache for 10 minutes (news feeds)
def get_market_news() -> List[Dict]:
"""
 Fetch market news from RSS feeds with error handling

 Returns:
 list: List of news articles
"""
 try:
 # Yahoo Finance RSS feed
 feed_url ="https://feeds.finance.yahoo.com/rss/2.0/headline"
 feed = feedparser.parse(feed_url)

 articles = []
 for entry in feed.entries[:15]: # Get top 15 articles
 articles.append({
'title': entry.title,
'link': entry.link,
'published': entry.published,
'summary': entry.get('summary', entry.title)
 })

 # Track cache timestamp
 set_cache_timestamp("market_news")
 return articles

 except Exception as e:
 st.warning(f"Error fetching news: {str(e)}")
 # Return fallback news
 return [{
'title':'Market News Unavailable',
'link':'#',
'published': datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z'),
'summary':'Unable to fetch current market news. Please check your internet connection.'
 }]

@st.cache_data(ttl=3600, show_spinner=False) # Cache for 1 hour (validation doesn't change often)
def validate_ticker(symbol: str) -> bool:
"""
 Validate if a ticker symbol exists and has data

 Args:
 symbol (str): Stock ticker symbol

 Returns:
 bool: True if valid ticker, False otherwise
"""
 try:
 if not symbol or not isinstance(symbol, str):
 return False

 symbol = symbol.upper().strip()
 ticker = yf.Ticker(symbol)

 # Try to get basic info
 info = ticker.info
 if not info or'symbol' not in info:
 return False

 # Try to get recent data
 data = ticker.history(period="5d")
 return not data.empty

 except Exception:
 return False

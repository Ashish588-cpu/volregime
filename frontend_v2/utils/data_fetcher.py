"""
Data fetching utilities for VolRegime frontend
Handles real-time market data using yfinance and other APIs
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple

# Cache data for 5 minutes to avoid excessive API calls
@st.cache_data(ttl=300)
def get_market_data(symbols: List[str]) -> Dict:
    """
    Fetch real-time market data for given symbols
    
    Args:
        symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    
    Returns:
        Dictionary with market data for each symbol
    """
    try:
        # Fetch data for all symbols at once
        tickers = yf.Tickers(' '.join(symbols))
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    market_data[symbol] = {
                        'name': info.get('longName', symbol),
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                        'market_cap': info.get('marketCap', 0),
                        'last_updated': datetime.now().strftime("%H:%M:%S")
                    }
                else:
                    # Fallback for symbols with no historical data
                    market_data[symbol] = {
                        'name': symbol,
                        'price': 0,
                        'change': 0,
                        'change_percent': 0,
                        'volume': 0,
                        'market_cap': 0,
                        'last_updated': datetime.now().strftime("%H:%M:%S")
                    }
                    
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return market_data
        
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_major_indices() -> Dict:
    """Get data for major market indices"""
    indices = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'Dow Jones',
        '^VIX': 'VIX',
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum'
    }
    
    return get_market_data(list(indices.keys()))

@st.cache_data(ttl=300)
def get_market_movers(limit: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Get top gainers and losers from popular stocks
    
    Args:
        limit: Number of stocks to return for each category
    
    Returns:
        Tuple of (gainers, losers) lists
    """
    # Popular stocks to check
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT'
    ]
    
    market_data = get_market_data(popular_stocks)
    
    # Sort by change percentage
    stocks_with_change = []
    for symbol, data in market_data.items():
        if data['change_percent'] != 0:  # Filter out stocks with no change data
            stocks_with_change.append({
                'symbol': symbol,
                'name': data['name'],
                'price': data['price'],
                'change_percent': data['change_percent'],
                'change': data['change']
            })
    
    # Sort and get top gainers and losers
    sorted_stocks = sorted(stocks_with_change, key=lambda x: x['change_percent'], reverse=True)
    gainers = sorted_stocks[:limit]
    losers = sorted_stocks[-limit:]
    
    return gainers, losers

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_history(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """
    Get historical stock data for charting
    
    Args:
        symbol: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        DataFrame with historical price data
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching history for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol: str) -> Dict:
    """
    Get detailed information about a stock
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'description': info.get('longBusinessSummary', 'No description available')
        }
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {str(e)}")
        return {}

def get_market_status() -> Dict:
    """
    Check if markets are currently open
    
    Returns:
        Dictionary with market status information
    """
    try:
        # US market hours (9:30 AM - 4:00 PM ET)
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Check if it's a weekday
        is_weekday = now_et.weekday() < 5
        
        # Check if it's within market hours
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = market_open <= now_et <= market_close
        
        is_open = is_weekday and is_market_hours
        
        return {
            'is_open': is_open,
            'timestamp': now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            'next_open': 'Next trading day 9:30 AM ET' if not is_open else 'Open now',
            'timezone': 'US/Eastern'
        }
    except Exception as e:
        return {
            'is_open': False,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'next_open': 'Unknown',
            'timezone': 'Local'
        }

def format_currency(value: float) -> str:
    """Format a number as currency"""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a number as percentage with color"""
    if value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

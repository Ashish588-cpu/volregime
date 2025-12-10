"""
Enhanced data utilities for VolRegime financial analytics
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Optional

@st.cache_data(ttl=300)
def get_market_indices() -> Dict:
    """Get major market indices data"""
    symbols = ['SPY', 'QQQ', 'BTC-USD', 'ETH-USD', '^VIX', '^DJI', '^GSPC', '^IXIC']
    data = {}
    
    try:
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            ticker = tickers.tickers[symbol]
            info = ticker.info
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0
                
                data[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_percent': change_pct,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                    'name': info.get('shortName', symbol)
                }
    except Exception as e:
        st.warning(f"Error fetching market data: {e}")
        
    return data

@st.cache_data(ttl=600)
def get_stock_data(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Get stock price data with indicators"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return pd.DataFrame()
            
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(window=14).mean()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol: str) -> Dict:
    """Get detailed stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'name': info.get('shortName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'beta': info.get('beta', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'description': info.get('longBusinessSummary', ''),
            'employees': info.get('fullTimeEmployees', 0),
            'website': info.get('website', ''),
            'previous_close': info.get('previousClose', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0)
        }
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {e}")
        return {}

@st.cache_data(ttl=900)
def get_sector_performance() -> Dict:
    """Get sector ETF performance"""
    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financial',
        'XLV': 'Healthcare', 
        'XLE': 'Energy',
        'XLI': 'Industrial',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials'
    }
    
    data = {}
    try:
        symbols = list(sector_etfs.keys())
        tickers = yf.Tickers(' '.join(symbols))
        
        for symbol, sector in sector_etfs.items():
            ticker = tickers.tickers[symbol]
            hist = ticker.history(period='1d')
            info = ticker.info
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
                
                data[sector] = {
                    'symbol': symbol,
                    'price': current_price,
                    'change_percent': change_pct
                }
    except Exception as e:
        st.warning(f"Error fetching sector data: {e}")
        
    return data

def calculate_portfolio_metrics(positions: List[Dict]) -> Dict:
    """Calculate portfolio metrics"""
    if not positions:
        return {}
        
    total_value = sum(pos['quantity'] * pos['price'] for pos in positions)
    total_cost = sum(pos['quantity'] * pos['cost_basis'] for pos in positions)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    # Calculate weights
    for pos in positions:
        pos['weight'] = (pos['quantity'] * pos['price']) / total_value if total_value > 0 else 0
        pos['pnl'] = pos['quantity'] * (pos['price'] - pos['cost_basis'])
        pos['pnl_pct'] = ((pos['price'] - pos['cost_basis']) / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'positions': positions
    }

def calculate_volatility(prices: pd.Series, window: int = 30) -> float:
    """Calculate realized volatility"""
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.iloc[-1] if not volatility.empty else 0

def calculate_drawdown(prices: pd.Series) -> Tuple[float, pd.Series]:
    """Calculate maximum drawdown and drawdown series"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

def calculate_realized_volatility(stock_data, window_days):
    """Calculate realized volatility over a rolling window"""
    try:
        returns = stock_data['Close'].pct_change().dropna()
        if len(returns) < window_days:
            return 0.0

        # Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window=window_days).std() * (252**0.5) * 100

        # Return the most recent volatility value
        return rolling_vol.iloc[-1] if not rolling_vol.empty else 0.0
    except Exception:
        return 0.0

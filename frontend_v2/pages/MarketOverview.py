"""
Market Overview page for VolRegime Financial Analytics Platform
Converted from Lovable React frontend to Streamlit
Tabbed interface for browsing 100+ assets across different asset classes
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.cards import metric_card
from components.loading import skeleton_card, inject_shimmer_css
from utils.data_core import get_stock_data, get_stock_info, get_current_price, handle_api_error


# ============================================================================
# ASSET LISTS BY CATEGORY
# ============================================================================

US_STOCKS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","V","JPM",
             "UNH","HD","WMT","MA","PG","XOM","CVX","LLY","BAC","ABBV"]

CRYPTO = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
          "ADA-USD","DOGE-USD","AVAX-USD","DOT-USD","MATIC-USD"]

FUTURES = [
    {"ticker":"ES=F","name":"S&P 500 Futures"},
    {"ticker":"NQ=F","name":"NASDAQ Futures"},
    {"ticker":"YM=F","name":"Dow Futures"},
    {"ticker":"RTY=F","name":"Russell Futures"},
    {"ticker":"GC=F","name":"Gold Futures"},
    {"ticker":"SI=F","name":"Silver Futures"},
    {"ticker":"CL=F","name":"Crude Oil WTI"},
    {"ticker":"NG=F","name":"Natural Gas"}
]

BONDS = ["TLT","IEF","SHY","AGG","BND","LQD","HYG","JNK"]

COMMODITIES = [
    {"ticker":"GLD","name":"Gold ETF"},
    {"ticker":"SLV","name":"Silver ETF"},
    {"ticker":"USO","name":"Oil ETF"},
    {"ticker":"UNG","name":"Natural Gas ETF"},
    {"ticker":"WEAT","name":"Wheat ETF"},
    {"ticker":"CORN","name":"Corn ETF"},
    {"ticker":"DBA","name":"Agriculture ETF"},
    {"ticker":"CPER","name":"Copper ETF"}
]


def load_styles():
    """Load CSS styles for Market Overview page"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-family:'Poppins', sans-serif;
        font-weight: 500;
        padding: 0 16px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        color: #f8fafc;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }

    /* Asset table styling */
    .asset-table {
        font-family:'Poppins', sans-serif;
    }

    /* Stat cards */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)


def format_market_cap(value):
    """Format market cap with K/M/B/T suffixes"""
    if value is None or value == 0:
        return"N/A"
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    return f"${value:.2f}"


def format_volume(value):
    """Format volume with K/M/B suffixes"""
    if value is None or value == 0:
        return"N/A"
    if value >= 1e9:
        return f"{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    return f"{value:.0f}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_asset_data(tickers: list, asset_type: str ="stock") -> pd.DataFrame:
    """Fetch data for a list of tickers and return as DataFrame"""
    rows = []

    for ticker in tickers:
        # Handle dict format for futures/commodities
        if isinstance(ticker, dict):
            symbol = ticker["ticker"]
            display_name = ticker["name"]
        else:
            symbol = ticker
            display_name = ticker

        try:
            # Get historical data for price and change
            data = get_stock_data(symbol, period="5d", show_errors=False)

            if data is None or data.empty or len(data) < 2:
                rows.append({
                    "Ticker": display_name,
                    "Price":"N/A",
                    "Change":"N/A",
                    "Change %":"N/A",
                    "Volume":"N/A",
                    "Market Cap":"N/A",
                    "_symbol": symbol,
                    "_change_pct": 0
                })
                continue

            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
            volume = data['Volume'].iloc[-1] if'Volume' in data.columns else 0

            # Get market cap for stocks
            market_cap = None
            if asset_type =="stock":
                info = get_stock_info(symbol, show_errors=False)
                if info and'marketCap' in info:
                    market_cap = info['marketCap']

            rows.append({
                "Ticker": display_name,
                "Price": f"${current_price:,.2f}",
                "Change": f"${change:+,.2f}",
                "Change %": f"{change_pct:+.2f}%",
                "Volume": format_volume(volume),
                "Market Cap": format_market_cap(market_cap),
                "_symbol": symbol,
                "_change_pct": change_pct
            })

        except Exception:
            rows.append({
                "Ticker": display_name,
                "Price":"N/A",
                "Change":"N/A",
                "Change %":"N/A",
                "Volume":"N/A",
                "Market Cap":"N/A",
                "_symbol": symbol,
                "_change_pct": 0
            })

    return pd.DataFrame(rows)


def render_asset_table(tickers, asset_type: str, show_market_cap: bool, tab_key: str):
    """Render an asset table with loading skeleton"""

    # Show loading skeleton
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 2rem; text-align: center;">
            <div class="shimmer" style="height: 300px; border-radius: 8px;"></div>
            <p style="color: #64748b; margin-top: 1rem;">Loading asset data...</p>
        </div>
        """, unsafe_allow_html=True)

    try:
        # Fetch data
        df = fetch_asset_data(tickers, asset_type)
        loading_placeholder.empty()

        if df.empty:
            st.warning("No data available for this asset class.")
            return df

        # Display styled dataframe
        display_df = df[["Ticker","Price","Change","Change %","Volume"] + (["Market Cap"] if show_market_cap else [])]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(500, len(display_df) * 40 + 40),
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Change": st.column_config.TextColumn("Change", width="small"),
                "Change %": st.column_config.TextColumn("Change %", width="small"),
                "Volume": st.column_config.TextColumn("Volume", width="small"),
                "Market Cap": st.column_config.TextColumn("Market Cap", width="medium") if show_market_cap else None
            }
        )

        return df

    except Exception as e:
        loading_placeholder.empty()
        st.markdown(handle_api_error(
            "Failed to load asset data. Please try again."
        ), unsafe_allow_html=True)
        return pd.DataFrame()


def render_gradient_hero(title: str, subtitle: str):
    """Render gradient hero header"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); padding: 48px; border-radius: 18px; text-align: center; margin-bottom: 32px; box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);">
        <h1 style="font-family:'Poppins', sans-serif; font-size: 48px; font-weight: 700; color: #ffffff; margin: 0 0 12px 0; text-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            {title}
        </h1>
        <p style="font-family:'Poppins', sans-serif; font-size: 20px; font-weight: 500; color: rgba(255,255,255,0.95); margin: 0;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_summary_stats(df: pd.DataFrame):
    """Render summary statistics cards"""
    if df.empty:
        return

    # Calculate stats
    total_assets = len(df)
    gainers = len(df[df['_change_pct'] > 0])
    losers = len(df[df['_change_pct'] < 0])
    avg_change = df['_change_pct'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Assets</div>
            <div class="stat-value">{total_assets}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Gainers</div>
            <div class="stat-value" style="color: #10b981;">{gainers}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Losers</div>
            <div class="stat-value" style="color: #ef4444;">{losers}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_color ="#10b981" if avg_change >= 0 else"#ef4444"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Avg Change</div>
            <div class="stat-value" style="color: {avg_color};">{avg_change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)


def show():
    """Display the market overview page with tabbed interface"""

    load_styles()
    inject_shimmer_css()

    # Back button at the top left
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="market_back_to_home"):
            st.session_state.current_page ="Home"
            st.rerun()

    # Hero section
    render_gradient_hero("Market Overview","Browse 100+ assets across stocks, crypto, futures, bonds & commodities")

    # Tabbed interface
    tabs = st.tabs([" US Stocks","🪙 Crypto"," Futures"," Bonds"," Commodities"])

    # TAB 1: US Stocks
    with tabs[0]:
        st.markdown("### Top 20 US Stocks by Market Cap")
        df = render_asset_table(US_STOCKS,"stock", True,"stocks")
        if not df.empty:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_summary_stats(df)

    # TAB 2: Crypto
    with tabs[1]:
        st.markdown("### Top 10 Cryptocurrencies")
        df = render_asset_table(CRYPTO,"crypto", True,"crypto")
        if not df.empty:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_summary_stats(df)

    # TAB 3: Futures
    with tabs[2]:
        st.markdown("### Major Futures Contracts")
        df = render_asset_table(FUTURES,"futures", False,"futures")
        if not df.empty:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_summary_stats(df)

    # TAB 4: Bonds
    with tabs[3]:
        st.markdown("### Bond ETFs")
        st.caption("Individual bonds not available via yfinance - showing popular bond ETFs instead")
        df = render_asset_table(BONDS,"bond", False,"bonds")
        if not df.empty:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_summary_stats(df)

    # TAB 5: Commodities
    with tabs[4]:
        st.markdown("### Commodities")
        df = render_asset_table(COMMODITIES,"commodity", False,"commodities")
        if not df.empty:
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_summary_stats(df)

    # Copyright footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #1e293b;
    ">
        <p>2025 || Ashish Dahal || copy right reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ =="__main__":
    show()

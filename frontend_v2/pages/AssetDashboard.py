"""
Asset Dashboard page for VolRegime Financial Analytics Platform
Deep analysis of individual assets with real-time data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.cards import metric_card, info_card, status_card
from components.loading import skeleton_card, skeleton_chart, inject_shimmer_css
from components.market_viz import create_multi_panel_chart, create_risk_heatmap_figure
from utils.data_core import (
    get_stock_data, get_current_price, get_stock_info,
    calculate_technical_indicators, get_trend_classification, validate_ticker,
    handle_api_error
)
from utils.market_augmentation import augment_asset_data, get_augmented_summary


# ============================================================================
# WATCHLIST FUNCTIONS
# ============================================================================

def init_watchlist():
    """Initialize watchlist in session state"""
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []


def add_to_watchlist(ticker: str):
    """Add ticker to watchlist"""
    init_watchlist()
    ticker = ticker.upper().strip()
    if ticker and ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)


def remove_from_watchlist(ticker: str):
    """Remove ticker from watchlist"""
    init_watchlist()
    ticker = ticker.upper().strip()
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)


def is_in_watchlist(ticker: str) -> bool:
    """Check if ticker is in watchlist"""
    init_watchlist()
    return ticker.upper().strip() in st.session_state.watchlist


def toggle_watchlist(ticker: str):
    """Toggle ticker in/out of watchlist"""
    if is_in_watchlist(ticker):
        remove_from_watchlist(ticker)
    else:
        add_to_watchlist(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def get_watchlist_prices(tickers: tuple) -> dict:
    """Fetch current prices for watchlist tickers (cached for 1 min)"""
    prices = {}
    for ticker in tickers:
        try:
            data = get_stock_data(ticker, period="5d", show_errors=False)
            if data is not None and not data.empty and len(data) >= 2:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change_pct = ((current - prev) / prev) * 100 if prev > 0 else 0
                prices[ticker] = {
                    'price': current,
                    'change_pct': change_pct
                }
            else:
                prices[ticker] = {'price': None, 'change_pct': 0}
        except Exception:
            prices[ticker] = {'price': None, 'change_pct': 0}
    return prices


def create_candlestick_chart(data: pd.DataFrame, ticker: str, period: str) -> go.Figure:
    """Create professional candlestick chart with technical indicators"""

    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker,
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444'
    ))

    # Add moving averages
    if 'SMA_20' in data.columns and not data['SMA_20'].isna().all():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#f59e0b', width=2),
            opacity=0.8
        ))

    if 'SMA_50' in data.columns and not data['SMA_50'].isna().all():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#3b82f6', width=2),
            opacity=0.8
        ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} - {period.upper()} Price Chart",
            font=dict(family='Poppins, sans-serif', size=20, color='#f8fafc')
        ),
        yaxis_title="Price ($)",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#0f172a',
        font=dict(family='Poppins, sans-serif', color='#f8fafc'),
        xaxis=dict(gridcolor='#334155', showgrid=True),
        yaxis=dict(gridcolor='#334155', showgrid=True),
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='#334155',
            borderwidth=1
        )
    )

    return fig

def create_technical_indicators_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create technical indicators chart with RSI and MACD

    Args:
        data (pd.DataFrame): Stock data with calculated technical indicators
        ticker (str): Stock ticker symbol

    Returns:
        go.Figure: Plotly figure with 2-row subplot
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('RSI', 'MACD'),
        row_heights=[0.3, 0.7]
    )

    # RSI - Relative Strength Index
    if 'RSI' in data.columns and not data['RSI'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='#8b5cf6')),
            row=1, col=1
        )
        # Overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

    # MACD - Moving Average Convergence Divergence
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']) and not data['MACD'].isna().all():
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#06b6d4')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='#f59e0b')),
            row=2, col=1
        )

    fig.update_layout(
        title=dict(
            text=f"{ticker} - Technical Indicators",
            font=dict(family='Poppins, sans-serif', size=18, color='#f8fafc')
        ),
        height=400,
        template="plotly_dark",
        paper_bgcolor='#1e293b',
        plot_bgcolor='#0f172a',
        font=dict(family='Poppins, sans-serif', color='#f8fafc'),
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155'),
        xaxis2=dict(gridcolor='#334155'),
        yaxis2=dict(gridcolor='#334155')
    )

    return fig


def create_drawdown_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create drawdown chart showing peak-to-trough declines

    Args:
        data (pd.DataFrame): Stock price data
        ticker (str): Stock ticker symbol

    Returns:
        go.Figure: Plotly figure showing drawdown over time
    """
    # Calculate running maximum (peak)
    running_max = data['Close'].cummax()

    # Calculate drawdown as percentage from peak
    drawdown = (data['Close'] - running_max) / running_max * 100

    # Create figure
    fig = go.Figure()

    # Add drawdown area chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ef4444', width=0),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.3)'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#1e293b", opacity=0.5, line_width=1)

    # Calculate max drawdown
    max_drawdown = drawdown.min()

    fig.update_layout(
        title=f"{ticker} - Drawdown Analysis (Max DD: {max_drawdown:.2f}%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400,
        template="plotly_white",
        font=dict(family='Poppins, sans-serif'),
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')

    return fig


def create_returns_distribution_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create returns distribution histogram with normal curve overlay

    Args:
        data (pd.DataFrame): Stock price data
        ticker (str): Stock ticker symbol

    Returns:
        go.Figure: Plotly figure showing returns distribution
    """
    # Calculate daily returns
    returns = data['Close'].pct_change().dropna() * 100  # Convert to percentage

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Daily Returns',
        marker=dict(
            color='#4f8df9',
            line=dict(color='#3d7be6', width=1)
        ),
        opacity=0.7
    ))

    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()

    # Add mean line
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="#059669",
        line_width=2,
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"{ticker} - Daily Returns Distribution (μ={mean_return:.2f}%, σ={std_return:.2f}%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white",
        font=dict(family='Poppins, sans-serif'),
        showlegend=False,
        bargap=0.05
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')

    return fig


def create_volatility_cone_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create volatility cone showing realized volatility at different time horizons

    Args:
        data (pd.DataFrame): Stock price data
        ticker (str): Stock ticker symbol

    Returns:
        go.Figure: Plotly figure showing volatility cone
    """
    # Calculate returns
    returns = data['Close'].pct_change().dropna()

    # Define windows for volatility calculation (trading days)
    windows = [5, 10, 20, 30, 60, 90, 120]
    window_labels = ['5d', '10d', '20d', '1m', '3m', '6m', '~1y']

    # Calculate realized volatility for each window
    volatilities = []
    for window in windows:
        if len(returns) >= window:
            vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized
            volatilities.append({
                'window': window,
                'label': window_labels[windows.index(window)],
                'current': vol.iloc[-1],
                'min': vol.min(),
                'max': vol.max(),
                'median': vol.median(),
                'q25': vol.quantile(0.25),
                'q75': vol.quantile(0.75)
            })

    if not volatilities:
        return None

    # Create volatility cone figure
    fig = go.Figure()

    # Extract data for plotting
    labels = [v['label'] for v in volatilities]
    current_vols = [v['current'] for v in volatilities]
    max_vols = [v['max'] for v in volatilities]
    q75_vols = [v['q75'] for v in volatilities]
    median_vols = [v['median'] for v in volatilities]
    q25_vols = [v['q25'] for v in volatilities]
    min_vols = [v['min'] for v in volatilities]

    # Add cone bands
    fig.add_trace(go.Scatter(
        x=labels, y=max_vols, name='Max', mode='lines',
        line=dict(color='rgba(239, 68, 68, 0.3)', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=q75_vols, name='75th Percentile', mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.4)', width=1),
        fill='tonexty', fillcolor='rgba(239, 68, 68, 0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=median_vols, name='Median', mode='lines+markers',
        line=dict(color='#3b82f6', width=2), marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=q25_vols, name='25th Percentile', mode='lines',
        line=dict(color='rgba(16, 185, 129, 0.4)', width=1),
        fill='tonexty', fillcolor='rgba(245, 158, 11, 0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=min_vols, name='Min', mode='lines',
        line=dict(color='rgba(16, 185, 129, 0.3)', width=1),
        fill='tonexty', fillcolor='rgba(16, 185, 129, 0.1)'
    ))

    # Add current volatility line
    fig.add_trace(go.Scatter(
        x=labels, y=current_vols, name='Current', mode='lines+markers',
        line=dict(color='#ef4444', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond')
    ))

    fig.update_layout(
        title=f"{ticker} - Volatility Cone (Annualized %)",
        xaxis_title="Time Horizon",
        yaxis_title="Volatility (%)",
        height=500,
        template="plotly_white",
        font=dict(family='Poppins, sans-serif'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        hovermode='x unified'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')

    return fig

def show():
    """Display the asset dashboard page"""

    # Initialize watchlist and selected ticker
    init_watchlist()
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = "AAPL"

    # Back button at the top left
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="back_to_home"):
            st.session_state.current_page = "Home"
            st.rerun()

    # Breadcrumb navigation
    st.markdown("""
    <div style="font-family: 'Poppins', sans-serif; color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
        <span style="color: #4f8df9;">Home</span> › <span style="color: #1e293b; font-weight: 500;">Asset Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero section with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4f8df9, #4752fa); padding: 42px; border-radius: 18px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(79, 141, 249, 0.3);">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 3rem; font-weight: 700; color: #000000; margin: 0 0 0.5rem 0;">Asset Dashboard</h1>
        <p style="font-family: 'Poppins', sans-serif; font-size: 1.2rem; color: #000000; margin: 0;">Deep analysis of individual assets with real-time data and technical indicators</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # WATCHLIST SECTION
    # =========================================================================
    st.markdown("### ⭐ My Watchlist")

    if st.session_state.watchlist:
        # Fetch prices for watchlist (use tuple for caching)
        watchlist_prices = get_watchlist_prices(tuple(st.session_state.watchlist))

        # Create horizontal scrollable watchlist
        st.markdown("""
        <style>
        .watchlist-container {
            display: flex;
            gap: 0.75rem;
            overflow-x: auto;
            padding: 0.5rem 0;
            margin-bottom: 1rem;
        }
        .watchlist-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            white-space: nowrap;
            transition: all 0.2s;
        }
        .watchlist-pill:hover {
            background: #dbeafe;
            border-color: #93c5fd;
        }
        .watchlist-ticker {
            font-weight: 600;
            color: #1e40af;
            font-size: 0.9rem;
        }
        .watchlist-price {
            color: #475569;
            font-size: 0.85rem;
        }
        .watchlist-change-up {
            color: #16a34a;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .watchlist-change-down {
            color: #dc2626;
            font-size: 0.8rem;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display watchlist as columns with buttons
        num_items = len(st.session_state.watchlist)
        cols_per_row = min(num_items, 5)
        watchlist_cols = st.columns(cols_per_row + 1)  # +1 for spacing

        for i, wl_ticker in enumerate(st.session_state.watchlist[:5]):  # Show max 5
            price_data = watchlist_prices.get(wl_ticker, {})
            price = price_data.get('price')
            change = price_data.get('change_pct', 0)

            with watchlist_cols[i]:
                # Price display
                price_str = f"${price:.2f}" if price else "N/A"
                change_color = "#16a34a" if change >= 0 else "#dc2626"
                change_icon = "▲" if change >= 0 else "▼"

                st.markdown(f"""
                <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-weight: 700; color: #1e40af; font-size: 1rem;">{wl_ticker}</div>
                    <div style="color: #475569; font-size: 0.85rem;">{price_str}</div>
                    <div style="color: {change_color}; font-size: 0.8rem; font-weight: 500;">{change_icon} {abs(change):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Button row
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📈", key=f"load_{wl_ticker}", help=f"Load {wl_ticker}"):
                        st.session_state.selected_ticker = wl_ticker
                        st.rerun()
                with btn_col2:
                    if st.button("✕", key=f"remove_{wl_ticker}", help=f"Remove {wl_ticker}"):
                        remove_from_watchlist(wl_ticker)
                        st.rerun()

        # Show count if more than 5
        if num_items > 5:
            with watchlist_cols[5]:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #64748b; font-size: 0.9rem;">
                    +{num_items - 5} more
                </div>
                """, unsafe_allow_html=True)

        # Expandable full list if > 5
        if num_items > 5:
            with st.expander(f"View all {num_items} watchlist items"):
                for wl_ticker in st.session_state.watchlist[5:]:
                    price_data = watchlist_prices.get(wl_ticker, {})
                    price = price_data.get('price')
                    change = price_data.get('change_pct', 0)
                    price_str = f"${price:.2f}" if price else "N/A"
                    change_color = "#16a34a" if change >= 0 else "#dc2626"

                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{wl_ticker}**")
                    with col2:
                        st.markdown(price_str)
                    with col3:
                        st.markdown(f"<span style='color: {change_color};'>{change:+.2f}%</span>", unsafe_allow_html=True)
                    with col4:
                        if st.button("✕", key=f"remove_exp_{wl_ticker}"):
                            remove_from_watchlist(wl_ticker)
                            st.rerun()
    else:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 1.5rem; text-align: center; color: #64748b;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⭐</div>
            <div>No tickers in watchlist yet.</div>
            <div style="font-size: 0.85rem; margin-top: 0.25rem;">Search and star tickers below to add them.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 
    # TICKER INPUT WITH STAR BUTTON
    # 
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker:",
            value=st.session_state.selected_ticker,
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol",
            key="ticker_input"
        ).upper()
        # Update selected ticker when manually changed
        if ticker != st.session_state.selected_ticker:
            st.session_state.selected_ticker = ticker

    with col2:
        period = st.selectbox(
            "Time Period:",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=3,  # Default to 3mo
            key="asset_period"
        )

    with col3:
        # Star button for watchlist
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
        if ticker:
            in_watchlist = is_in_watchlist(ticker)
            star_icon = "★" if in_watchlist else "☆"
            star_label = "Remove" if in_watchlist else "Add"
            star_color = "#3b82f6" if in_watchlist else "#94a3b8"

            if st.button(
                f"{star_icon} {star_label}",
                key="watchlist_toggle",
                help=f"{'Remove from' if in_watchlist else 'Add to'} watchlist",
                use_container_width=True
            ):
                toggle_watchlist(ticker)
                st.rerun()

    if ticker:
        # Validate ticker
        if not validate_ticker(ticker):
            st.markdown(handle_api_error(
                f"Invalid ticker symbol. Please enter a valid stock ticker.",
                ticker=ticker
            ), unsafe_allow_html=True)
            return

        # Inject shimmer CSS
        inject_shimmer_css()

        # Create placeholders for skeleton loading
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()

        # Show skeleton metrics while loading
        with metrics_placeholder.container():
            skeleton_cols = st.columns(4)
            for i in range(4):
                with skeleton_cols[i]:
                    st.markdown(skeleton_card(), unsafe_allow_html=True)

        # Show skeleton chart while loading
        chart_placeholder.markdown(skeleton_chart(height=400, message=f"Loading {ticker} chart..."), unsafe_allow_html=True)

        try:
            # Get stock data with error handling
            stock_data = get_stock_data(ticker, period=period, show_errors=False)
            current_price = get_current_price(ticker, show_errors=False)
            stock_info = get_stock_info(ticker, show_errors=False)

            # Clear skeleton placeholders
            metrics_placeholder.empty()
            chart_placeholder.empty()

            # Validate stock data before proceeding
            if stock_data is None or stock_data.empty:
                st.markdown(handle_api_error(
                    f"No data available for this ticker. It may be delisted or invalid.",
                    ticker=ticker
                ), unsafe_allow_html=True)
                return

            # Validate current price
            if current_price is None or current_price == 0:
                # Fallback to last close price from historical data
                current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0
                if current_price == 0:
                    st.markdown(handle_api_error(
                        "Unable to retrieve current price data.",
                        ticker=ticker
                    ), unsafe_allow_html=True)
                    return

            # Calculate metrics safely
            prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else stock_data['Close'].iloc[-1]
            if prev_close > 0:
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
            else:
                change = 0
                change_pct = 0

            # Display selected asset information at top
            if stock_info:
                asset_name = stock_info.get('longName', ticker)
                sector = stock_info.get('sector', 'N/A')
                industry = stock_info.get('industry', 'N/A')

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #0d0d10, #1a1a1d);
                    border: 1px solid #00eaff;
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(0,234,255,0.2);
                ">
                    <h2 style="
                        font-family: 'Space Grotesk', sans-serif;
                        color: #00eaff;
                        font-size: 1.75rem;
                        margin: 0 0 0.5rem 0;
                    ">{ticker} - {asset_name}</h2>
                    <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.95rem;">
                        <span style="color: #00eaff;">Sector:</span> {sector} &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color: #00eaff;">Industry:</span> {industry}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #0d0d10, #1a1a1d);
                    border: 1px solid #00eaff;
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    box-shadow: 0 0 20px rgba(0,234,255,0.2);
                ">
                    <h2 style="
                        font-family: 'Space Grotesk', sans-serif;
                        color: #00eaff;
                        font-size: 1.75rem;
                        margin: 0;
                    ">{ticker}</h2>
                </div>
                """, unsafe_allow_html=True)

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{change_pct:+.2f}%",
                    "positive" if change >= 0 else "negative"
                )

            with col2:
                volume = stock_data['Volume'].iloc[-1] if 'Volume' in stock_data.columns else 0
                metric_card(
                    "Volume",
                    f"{volume:,.0f}",
                    "Today's Volume"
                )

            with col3:
                if len(stock_data) > 20:
                    volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                    metric_card(
                        "Volatility",
                        f"{volatility:.2f}%",
                        "Annualized"
                    )

            with col4:
                if stock_info and 'marketCap' in stock_info:
                    market_cap = stock_info['marketCap']
                    if market_cap > 1e12:
                        cap_str = f"${market_cap/1e12:.2f}T"
                    elif market_cap > 1e9:
                        cap_str = f"${market_cap/1e9:.2f}B"
                    else:
                        cap_str = f"${market_cap/1e6:.2f}M"

                    metric_card(
                        "Market Cap",
                        cap_str,
                        "Market Capitalization"
                    )

            st.markdown("---")

            # Price chart
            st.markdown("### 📈 Price Chart")

            # Add technical indicators to data
            stock_data_with_indicators = calculate_technical_indicators(stock_data)

            # Create and display candlestick chart
            price_fig = create_candlestick_chart(stock_data_with_indicators, ticker, period)
            st.plotly_chart(price_fig, use_container_width=True)

            # Add spacing
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

            # Technical indicators chart
            st.markdown("### 📊 Technical Indicators")

            tech_fig = create_technical_indicators_chart(stock_data_with_indicators, ticker)
            st.plotly_chart(tech_fig, use_container_width=True)

            # Add spacing
            st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("---")

            # Advanced analytics section - Drawdown and Risk Metrics
            st.markdown("### 📉 Risk & Performance Analytics")

            col1, col2 = st.columns(2)

            with col1:
                # Drawdown chart
                drawdown_fig = create_drawdown_chart(stock_data, ticker)
                if drawdown_fig:
                    st.plotly_chart(drawdown_fig, use_container_width=True)

            with col2:
                # Returns distribution
                returns_fig = create_returns_distribution_chart(stock_data, ticker)
                if returns_fig:
                    st.plotly_chart(returns_fig, use_container_width=True)

            # Add spacing
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

            # Volatility cone
            st.markdown("### 📊 Volatility Analysis")
            vol_cone_fig = create_volatility_cone_chart(stock_data, ticker)
            if vol_cone_fig:
                st.plotly_chart(vol_cone_fig, use_container_width=True)

            # Add spacing
            st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
            st.markdown("---")

            # ================================================================
            # ADVANCED MULTI-PANEL ANALYSIS 
            # ================================================================
            st.markdown("###  Advanced Multi-Factor Analysis")

            with st.expander("**Multi-Panel Chart with Regime Detection**", expanded=False):
                st.markdown("""
                <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                            padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: #000000;">
                    <span style="font-weight: 700;">Quantitative Analysis</span> -
                    This chart shows price action, volatility regimes, and volume analysis with engineered features.
                </div>
                """, unsafe_allow_html=True)

                # main chart area with controls in right column
                chart_col, controls_col = st.columns([4, 1])

                # Controls in right sidebar
                with controls_col:
                    st.markdown("### Chart Controls")
                    show_ma = st.multiselect(
                        'Moving Averages',
                        options=[20, 50, 200],
                        default=[],
                        key='adv_ma_select'
                    )
                    show_earnings = st.checkbox('Show Earnings', value=False, key='adv_earnings')
                    show_regime = st.checkbox('Regime Bands', value=False, key='adv_regime')
                    show_macro = st.checkbox('Macro Correlations', value=False, key='adv_macro')

                # Generate augmented data and chart in main chart column
                with chart_col:
                    with st.spinner('Generating advanced analysis...'):
                        try:
                            import warnings
                            warnings.filterwarnings('ignore')

                            # Augment the data with all features
                            augmented_df = augment_asset_data(ticker, period=period, include_earnings=True)

                            if not augmented_df.empty:
                                # Create multi-panel chart
                                multi_fig = create_multi_panel_chart(
                                    df=augmented_df,
                                    symbol=ticker,
                                    show_ma=show_ma,
                                    show_earnings=show_earnings,
                                    show_regime_bands=show_regime,
                                    show_macro=show_macro,
                                    height=800
                                )

                                st.plotly_chart(multi_fig, use_container_width=True, config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'displaylogo': False
                                })

                                # Show current regime and key metrics
                                summary = get_augmented_summary(augmented_df)
                                metrics = summary.get('current_metrics', {})

                                st.markdown("#### Current Regime & Risk Metrics")

                                reg_col1, reg_col2, reg_col3, reg_col4, reg_col5 = st.columns(5)

                                with reg_col1:
                                    regime = metrics.get('regime', 'N/A')
                                    regime_colors = {
                                        'Low': 'Low', 'Medium': 'Med',
                                        'High': 'High', 'Extreme': 'Extreme'
                                    }
                                    st.metric("Volatility Regime", regime_colors.get(regime, 'N/A'))

                                with reg_col2:
                                    vol = metrics.get('realized_vol_20d', 0)
                                    if vol and not (vol != vol):
                                        st.metric("Realized Vol (20d)", f"{vol*100:.1f}%")
                                    else:
                                        st.metric("Realized Vol (20d)", "N/A")

                                with reg_col3:
                                    var = metrics.get('VaR_95', 0)
                                    if var and not (var != var):
                                        st.metric("VaR (95%)", f"{var*100:.2f}%")
                                    else:
                                        st.metric("VaR (95%)", "N/A")

                                with reg_col4:
                                    beta = metrics.get('beta', 1.0)
                                    if beta and not (beta != beta):
                                        st.metric("Beta (SPY)", f"{beta:.2f}")
                                    else:
                                        st.metric("Beta (SPY)", "N/A")

                                with reg_col5:
                                    momentum = metrics.get('momentum_score', 0.5)
                                    if momentum and not (momentum != momentum):
                                        st.metric("Momentum Score", f"{momentum:.2f}")
                                    else:
                                        st.metric("Momentum Score", "N/A")

                                # Risk Heatmap
                                st.markdown("#### Risk Factor Heatmap")
                                heatmap_fig = create_risk_heatmap_figure(augmented_df)
                                st.plotly_chart(heatmap_fig, use_container_width=True)

                            else:
                                st.warning("Could not generate advanced analysis for this ticker.")

                        except Exception as e:
                            st.error(f"Error generating advanced analysis: {str(e)}")

            st.markdown("---")

            # Additional metrics
            col1, col2 = st.columns(2)

            with col1:
                if 'RSI' in stock_data_with_indicators.columns:
                    current_rsi = stock_data_with_indicators['RSI'].iloc[-1]
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    metric_card(
                        "RSI",
                        f"{current_rsi:.2f}",
                        rsi_status
                    )

            with col2:
                if 'ATR' in stock_data_with_indicators.columns:
                    current_atr = stock_data_with_indicators['ATR'].iloc[-1]
                    metric_card(
                        "ATR",
                        f"${current_atr:.2f}",
                        "Average True Range"
                    )

            # Company information
            if stock_info:
                st.markdown("---")
                st.markdown("### Company Information")

                col1, col2 = st.columns(2)

                with col1:
                    if 'longName' in stock_info:
                        st.markdown(f"**Company:** {stock_info['longName']}")
                    if 'sector' in stock_info:
                        st.markdown(f"**Sector:** {stock_info['sector']}")
                    if 'industry' in stock_info:
                        st.markdown(f"**Industry:** {stock_info['industry']}")

                with col2:
                    if 'beta' in stock_info:
                        st.markdown(f"**Beta:** {stock_info['beta']:.2f}")
                    if 'trailingPE' in stock_info:
                        st.markdown(f"**P/E Ratio:** {stock_info['trailingPE']:.2f}")
                    if 'dividendYield' in stock_info and stock_info['dividendYield']:
                        st.markdown(f"**Dividend Yield:** {stock_info['dividendYield']*100:.2f}%")

                if 'longBusinessSummary' in stock_info:
                    st.markdown("**Business Summary:**")
                    st.markdown(stock_info['longBusinessSummary'][:500] + "..." if len(stock_info['longBusinessSummary']) > 500 else stock_info['longBusinessSummary'])

        except Exception as e:
            # Clear skeleton placeholders on error
            metrics_placeholder.empty()
            chart_placeholder.empty()
            # Catch-all error handler - ensures app never crashes
            st.markdown(handle_api_error(
                f"An unexpected error occurred while loading data. Please try again.",
                ticker=ticker
            ), unsafe_allow_html=True)
            st.info("💡 If this problem persists, try a different ticker symbol or check your internet connection.")

if __name__ == "__main__":
    show()


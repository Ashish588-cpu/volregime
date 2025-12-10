"""
Portfolio Tools page for VolRegime Financial Analytics Platform
Comprehensive portfolio management and analysis tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.cards import metric_card
from utils.data_core import get_current_price, get_stock_data, validate_ticker, handle_api_error

def calculate_portfolio_metrics(portfolio_df: pd.DataFrame) -> dict:
    """Calculate comprehensive portfolio metrics"""

    if portfolio_df.empty:
        return {}

    # Calculate total value and weights
    portfolio_df['Total Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
    total_portfolio_value = portfolio_df['Total Value'].sum()
    portfolio_df['Weight'] = portfolio_df['Total Value'] / total_portfolio_value

    # Calculate P&L
    portfolio_df['Total Cost'] = portfolio_df['Shares'] * portfolio_df['Cost Basis']
    portfolio_df['P&L'] = portfolio_df['Total Value'] - portfolio_df['Total Cost']
    portfolio_df['P&L %'] = (portfolio_df['P&L'] / portfolio_df['Total Cost']) * 100

    total_pnl = portfolio_df['P&L'].sum()
    total_cost = portfolio_df['Total Cost'].sum()
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

    # Calculate concentration risk (Herfindahl Index)
    hhi = (portfolio_df['Weight'] ** 2).sum()
    diversification_score = (1 - hhi) * 100 # Convert to 0-100 scale

    # Calculate portfolio volatility (simplified)
    avg_volatility = 0
    if len(portfolio_df) > 0:
        volatilities = []
        for ticker in portfolio_df['Ticker']:
            try:
                data = get_stock_data(ticker, period='3mo')
                if not data.empty:
                    vol = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    volatilities.append(vol)
            except:
                continue

        if volatilities:
            # Weighted average volatility (simplified, doesn't account for correlations)
            weights = portfolio_df['Weight'].values[:len(volatilities)]
            avg_volatility = np.average(volatilities, weights=weights)

    return {
        'total_value': total_portfolio_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'diversification_score': diversification_score,
        'avg_volatility': avg_volatility,
        'num_positions': len(portfolio_df)
    }

def create_allocation_chart(portfolio_df: pd.DataFrame) -> go.Figure:
    """Create portfolio allocation pie chart"""

    if portfolio_df.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Pie(
        labels=portfolio_df['Ticker'],
        values=portfolio_df['Total Value'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        )
    )])

    fig.update_layout(
        title="Portfolio Allocation",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    return fig


def create_correlation_heatmap(portfolio_df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap for portfolio holdings

    Args:
        portfolio_df (pd.DataFrame): Portfolio data with tickers

    Returns:
        go.Figure: Plotly heatmap showing correlations
    """
    if portfolio_df.empty or len(portfolio_df) < 2:
        return None

    try:
        # Fetch historical data for all tickers
        returns_data = {}

        for ticker in portfolio_df['Ticker'].unique():
            try:
                data = get_stock_data(ticker, period='6mo')
                if not data.empty:
                    returns_data[ticker] = data['Close'].pct_change().dropna()
            except:
                continue

        if len(returns_data) < 2:
            return None

        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Portfolio Correlation Matrix",
            xaxis_title="",
            yaxis_title="",
            height=500,
            template="plotly_white",
            font=dict(family='Poppins, sans-serif')
        )

        return fig

    except Exception as e:
        st.error(f"Error creating correlation heatmap: {e}")
        return None


def create_efficient_frontier(portfolio_df: pd.DataFrame) -> go.Figure:
    """
    Create efficient frontier plot for portfolio optimization

    Args:
        portfolio_df (pd.DataFrame): Portfolio data with tickers and weights

    Returns:
        go.Figure: Plotly scatter plot showing efficient frontier
    """
    if portfolio_df.empty or len(portfolio_df) < 2:
        return None

    try:
        # Fetch historical data for all tickers
        returns_data = {}

        for ticker in portfolio_df['Ticker'].unique():
            try:
                data = get_stock_data(ticker, period='1y')
                if not data.empty:
                    returns_data[ticker] = data['Close'].pct_change().dropna()
            except:
                continue

        if len(returns_data) < 2:
            return None

        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data)

        # Calculate expected returns and covariance
        mean_returns = returns_df.mean() * 252 # Annualized
        cov_matrix = returns_df.cov() * 252 # Annualized

        # Generate random portfolios for efficient frontier
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_array = []

        for i in range(num_portfolios):
            # Random weights
            weights = np.random.random(len(returns_df.columns))
            weights /= np.sum(weights)
            weights_array.append(weights)

            # Portfolio return and risk
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_std

            results[0,i] = portfolio_std
            results[1,i] = portfolio_return
            results[2,i] = sharpe_ratio

        # Calculate current portfolio metrics
        current_weights = portfolio_df['Weight'].values
        current_return = np.dot(current_weights, mean_returns)
        current_std = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))

        # Create scatter plot
        fig = go.Figure()

        # Add efficient frontier points
        fig.add_trace(go.Scatter(
            x=results[0,:],
            y=results[1,:],
            mode='markers',
            name='Simulated Portfolios',
            showlegend=False,
            marker=dict(
                size=3,
                color=results[2,:],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.6
            ),
            text=[f"Return: {r:.2%}<br>Risk: {s:.2%}<br>Sharpe: {sh:.2f}"
                  for r, s, sh in zip(results[1,:], results[0,:], results[2,:])],
            hovertemplate='%{text}<extra></extra>'
        ))

        # Add current portfolio point
        fig.add_trace(go.Scatter(
            x=[current_std],
            y=[current_return],
            mode='markers',
            name='Current Portfolio',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            text=f"Your Portfolio<br>Return: {current_return:.2%}<br>Risk: {current_std:.2%}",
            hovertemplate='%{text}<extra></extra>'
        ))

        # Find max Sharpe ratio portfolio
        max_sharpe_idx = results[2,:].argmax()
        max_sharpe_return = results[1, max_sharpe_idx]
        max_sharpe_std = results[0, max_sharpe_idx]

        fig.add_trace(go.Scatter(
            x=[max_sharpe_std],
            y=[max_sharpe_return],
            mode='markers',
            name='Max Sharpe Ratio',
            marker=dict(
                size=15,
                color='green',
                symbol='diamond',
                line=dict(color='darkgreen', width=2)
            ),
            text=f"Max Sharpe<br>Return: {max_sharpe_return:.2%}<br>Risk: {max_sharpe_std:.2%}",
            hovertemplate='%{text}<extra></extra>'
        ))

        fig.update_layout(
            title="Efficient Frontier - Portfolio Optimization",
            xaxis_title="Volatility (Risk) %",
            yaxis_title="Expected Return %",
            height=600,
            template="plotly_white",
            font=dict(family='Poppins, sans-serif'),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.9)"
            ),
            hovermode='closest'
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')

        return fig

    except Exception as e:
        st.error(f"Error creating efficient frontier: {e}")
        return None


def calculate_sharpe_ratio(portfolio_df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """
    Calculate Sharpe ratio and related metrics for portfolio

    Args:
        portfolio_df (pd.DataFrame): Portfolio data
        risk_free_rate (float): Annual risk-free rate (default: 2%)

    Returns:
        dict: Portfolio metrics including Sharpe ratio
    """
    if portfolio_df.empty:
        return None

    try:
        # Fetch historical data for all tickers
        returns_data = {}

        for ticker in portfolio_df['Ticker'].unique():
            try:
                data = get_stock_data(ticker, period='1y')
                if not data.empty:
                    returns_data[ticker] = data['Close'].pct_change().dropna()
            except:
                continue

        if not returns_data:
            return None

        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data)

        # Calculate portfolio returns
        weights = portfolio_df['Weight'].values
        portfolio_returns = returns_df.dot(weights)

        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

        # Calculate Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Calculate max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown
        }

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return None


# ============================================================================
# PORTFOLIO REGIME SIMULATOR
# ============================================================================

REGIME_COLORS = {
    'Low': {'bg':'#dcfce7','border':'#86efac','text':'#166534','chart':'#22c55e'},
    'Medium': {'bg':'#fef9c3','border':'#fde047','text':'#854d0e','chart':'#eab308'},
    'High': {'bg':'#fed7aa','border':'#fdba74','text':'#9a3412','chart':'#f97316'},
    'Extreme': {'bg':'#fecaca','border':'#fca5a5','text':'#991b1b','chart':'#ef4444'}
}


def classify_vix_regime(vix_value: float) -> str:
    """Classify VIX value into regime"""
    if vix_value < 15:
        return'Low'
    elif vix_value < 20:
        return'Medium'
    elif vix_value < 30:
        return'High'
    else:
        return'Extreme'


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_regime_performance(tickers: tuple, weights: tuple) -> dict:
    """
    Calculate portfolio performance across different volatility regimes

    Args:
        tickers: Tuple of ticker symbols
        weights: Tuple of portfolio weights

    Returns:
        dict with regime performance metrics
    """
    try:
        # Fetch 2 years of data for all tickers
        returns_data = {}
        for ticker in tickers:
            try:
                data = get_stock_data(ticker, period='2y', show_errors=False)
                if data is not None and not data.empty:
                    returns_data[ticker] = data['Close'].pct_change().dropna()
            except Exception:
                continue

        if len(returns_data) < 2:
            return None

        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty:
            return None

        # Fetch VIX data for same period
        vix_data = get_stock_data("^VIX", period='2y', show_errors=False)
        if vix_data is None or vix_data.empty:
            return None

        # Align dates
        common_dates = returns_df.index.intersection(vix_data.index)
        if len(common_dates) < 30: # Need at least 30 days
            return None

        returns_df = returns_df.loc[common_dates]
        vix_df = vix_data.loc[common_dates]

        # Calculate portfolio daily returns using weights
        weights_array = np.array(weights[:len(returns_df.columns)])
        weights_array = weights_array / weights_array.sum() # Normalize

        portfolio_returns = (returns_df * weights_array).sum(axis=1)

        # Classify each day into regime
        regimes = vix_df['Close'].apply(classify_vix_regime)

        # Calculate metrics per regime
        results = {}

        for regime in ['Low','Medium','High','Extreme']:
            regime_mask = regimes == regime
            regime_returns = portfolio_returns[regime_mask]

            if len(regime_returns) > 0:
                # Calculate metrics
                avg_daily_return = regime_returns.mean() * 100
                cumulative_return = ((1 + regime_returns).prod() - 1) * 100
                volatility = regime_returns.std() * np.sqrt(252) * 100

                # Sharpe ratio (annualized, assuming 2% risk-free rate)
                annual_return = avg_daily_return * 252
                sharpe = (annual_return - 2) / volatility if volatility > 0 else 0

                # Max drawdown
                cumulative = (1 + regime_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100

                # Days count
                days = len(regime_returns)
                pct_of_period = (days / len(portfolio_returns)) * 100

                results[regime] = {
                    'days': days,
                    'pct_of_period': pct_of_period,
                    'avg_daily_return': avg_daily_return,
                    'cumulative_return': cumulative_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'max_drawdown': max_drawdown
                }
            else:
                results[regime] = {
                    'days': 0,
                    'pct_of_period': 0,
                    'avg_daily_return': 0,
                    'cumulative_return': 0,
                    'volatility': 0,
                    'sharpe': 0,
                    'max_drawdown': 0
                }

        return results

    except Exception as e:
        return None


def render_regime_card(regime: str, metrics: dict):
    """Render a regime performance card"""
    colors = REGIME_COLORS[regime]

    change_color ='#16a34a' if metrics['cumulative_return'] >= 0 else'#dc2626'

    st.markdown(f"""
    <div style="background: {colors['bg']}; border: 2px solid {colors['border']}; border-radius: 12px; padding: 1.25rem; height: 100%;">
        <div style="font-weight: 700; color: {colors['text']}; font-size: 1rem; margin-bottom: 1rem; text-align: center;">
            {regime} Volatility
        </div>
        <div style="display: grid; gap: 0.5rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Days:</span>
                <span style="font-weight: 600; color: #1e293b;">{metrics['days']} ({metrics['pct_of_period']:.1f}%)</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Avg Daily:</span>
                <span style="font-weight: 600; color: {change_color};">{metrics['avg_daily_return']:+.3f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Cumulative:</span>
                <span style="font-weight: 700; color: {change_color};">{metrics['cumulative_return']:+.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Volatility:</span>
                <span style="font-weight: 600; color: #1e293b;">{metrics['volatility']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Max DD:</span>
                <span style="font-weight: 600; color: #dc2626;">{metrics['max_drawdown']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span style="color: #64748b;">Sharpe:</span>
                <span style="font-weight: 600; color: #1e293b;">{metrics['sharpe']:.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_regime_comparison_chart(regime_results: dict) -> go.Figure:
    """Create bar chart comparing returns across regimes"""

    regimes = ['Low','Medium','High','Extreme']
    returns = [regime_results[r]['cumulative_return'] for r in regimes]
    colors = [REGIME_COLORS[r]['chart'] for r in regimes]

    fig = go.Figure(data=[
        go.Bar(
            x=regimes,
            y=returns,
            marker_color=colors,
            text=[f"{r:+.1f}%" for r in returns],
            textposition='outside',
            hovertemplate='<b>%{x} Regime</b><br>Cumulative Return: %{y:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Portfolio Cumulative Returns by Volatility Regime",
        xaxis_title="Volatility Regime",
        yaxis_title="Cumulative Return (%)",
        height=400,
        template="plotly_white",
        font=dict(family='Poppins, sans-serif'),
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinecolor='#94a3b8', zerolinewidth=2)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')

    return fig


def generate_regime_insight(regime_results: dict) -> str:
    """Generate custom insight based on regime performance"""

    low_return = regime_results['Low']['cumulative_return']
    high_return = regime_results['High']['cumulative_return']
    extreme_return = regime_results['Extreme']['cumulative_return']

    # Check for vulnerability in extreme volatility
    if extreme_return < -10:
        return"️ **Warning:** Your portfolio struggles significantly in extreme volatility periods. Consider adding defensive assets like bonds (TLT, AGG), gold (GLD), or increasing cash allocation to buffer against market crashes."

    # Check if portfolio does better in high volatility
    if high_return > low_return and high_return > 0:
        return" **Observation:** Your portfolio actually performs better in high volatility environments. This suggests a growth-heavy or momentum-oriented allocation that benefits from market swings."

    # Check for balanced performance
    returns = [low_return, regime_results['Medium']['cumulative_return'], high_return, extreme_return]
    return_range = max(returns) - min(returns)

    if return_range < 15:
        return" **Good News:** Your portfolio shows relatively balanced performance across all regimes. This indicates good diversification and regime resilience."

    # Default insight
    if low_return > high_return:
        return" **Insight:** Your portfolio performs best in calm markets but weakens during volatility spikes. This is typical for equity-heavy portfolios. Consider adding volatility hedges if you're concerned about drawdowns."

    return" **Analysis:** Your portfolio shows varying performance across regimes. Review the metrics above to understand your exposure to different market conditions."


def show():
    """Display the portfolio tools page"""

    # Back button at the top left
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="portfolio_back_to_home"):
            st.session_state.current_page ="Home"
            st.rerun()

    # Breadcrumb navigation
    st.markdown("""
    <div style="font-family:'Poppins', sans-serif; color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
        <span style="color: #4f8df9;">Home</span> › <span style="color: #1e293b; font-weight: 500;">Portfolio Tools</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero section with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4f8df9, #4752fa); padding: 42px; border-radius: 18px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(79, 141, 249, 0.3);">
        <h1 style="font-family:'Poppins', sans-serif; font-size: 3rem; font-weight: 700; color: #000000; margin: 0 0 0.5rem 0;">Portfolio Tools</h1>
        <p style="font-family:'Poppins', sans-serif; font-size: 1.2rem; color: #000000; margin: 0;">Comprehensive portfolio management and risk analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for portfolio
    if'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=[
            'Ticker','Shares','Cost Basis','Current Price','Total Value','P&L','P&L %'
        ])

    # Portfolio input section
    st.markdown("### Portfolio Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Add Position")

        # Search mode selector
        search_mode = st.radio(
            "Search by:",
            ["Ticker Symbol","Company Name"],
            horizontal=True,
            label_visibility="collapsed"
        )

        input_col1, input_col2, input_col3, input_col4 = st.columns(4)

        with input_col1:
            if search_mode =="Company Name":
                company_search = st.text_input("Company Name", placeholder="Apple Inc")
                ticker =""

                if company_search:
                    import yfinance as yf
                    try:
                        # try common exchange suffixes
                        test_ticker = company_search.upper().replace("","")
                        ticker_obj = yf.Ticker(test_ticker)
                        info = ticker_obj.info

                        if'symbol' in info:
                            ticker = info['symbol']
                            st.success(f"Found: {info.get('longName', ticker)} ({ticker})")
                        else:
                            st.warning("Could not find ticker. Try entering the symbol directly.")
                    except:
                        st.warning("Could not find company. Try entering the ticker symbol directly.")
            else:
                ticker = st.text_input("Ticker", placeholder="AAPL").upper()

        with input_col2:
            shares = st.number_input("Shares", min_value=0.0, step=1.0, format="%.2f")

        with input_col3:
            cost_basis = st.number_input("Cost Basis ($)", min_value=0.0, step=0.01, format="%.2f")

        with input_col4:
            st.markdown("<br>", unsafe_allow_html=True) # Spacing
            if st.button("Add Position", use_container_width=True):
                if ticker and shares > 0 and cost_basis > 0:
                    if validate_ticker(ticker):
                        try:
                            current_price = get_current_price(ticker, show_errors=False)

                            # Validate current price was retrieved
                            if current_price is None or current_price == 0:
                                st.markdown(handle_api_error(
                                    "Unable to fetch current price. Please try again.",
                                    ticker=ticker
                                ), unsafe_allow_html=True)
                            else:
                                # Check if ticker already exists
                                if ticker in st.session_state.portfolio['Ticker'].values:
                                    # Update existing position
                                    idx = st.session_state.portfolio[st.session_state.portfolio['Ticker'] == ticker].index[0]
                                    existing_shares = st.session_state.portfolio.loc[idx,'Shares']
                                    existing_cost = st.session_state.portfolio.loc[idx,'Cost Basis'] * existing_shares

                                    new_total_shares = existing_shares + shares
                                    new_avg_cost = (existing_cost + (shares * cost_basis)) / new_total_shares

                                    st.session_state.portfolio.loc[idx,'Shares'] = new_total_shares
                                    st.session_state.portfolio.loc[idx,'Cost Basis'] = new_avg_cost
                                    st.session_state.portfolio.loc[idx,'Current Price'] = current_price
                                else:
                                    # Add new position
                                    new_row = pd.DataFrame({
                                        'Ticker': [ticker],
                                        'Shares': [shares],
                                        'Cost Basis': [cost_basis],
                                        'Current Price': [current_price],
                                        'Total Value': [0],
                                        'P&L': [0],
                                        'P&L %': [0]
                                    })
                                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)

                                st.success(f"Added {shares} shares of {ticker}")
                                st.rerun()

                        except Exception as e:
                            st.markdown(handle_api_error(
                                f"Error adding position. Please try again.",
                                ticker=ticker
                            ), unsafe_allow_html=True)
                    else:
                        st.markdown(handle_api_error(
                            "Invalid ticker symbol. Please enter a valid stock ticker.",
                            ticker=ticker
                        ), unsafe_allow_html=True)
                else:
                    st.warning("Please fill in all fields with valid values")

    with col2:
        st.markdown("#### Quick Actions")

        if st.button(" Refresh All Prices", use_container_width=True):
            if not st.session_state.portfolio.empty:
                with st.spinner("Updating prices..."):
                    failed_tickers = []
                    for idx, row in st.session_state.portfolio.iterrows():
                        try:
                            current_price = get_current_price(row['Ticker'], show_errors=False)
                            if current_price is not None and current_price > 0:
                                st.session_state.portfolio.loc[idx,'Current Price'] = current_price
                            else:
                                failed_tickers.append(row['Ticker'])
                        except Exception:
                            failed_tickers.append(row['Ticker'])
                            continue

                    if failed_tickers:
                        st.warning(f"Could not update prices for: {','.join(failed_tickers)}")
                    else:
                        st.success("All prices updated!")
                    st.rerun()
            else:
                st.info("No positions to update")

        if st.button("️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = pd.DataFrame(columns=[
                'Ticker','Shares','Cost Basis','Current Price','Total Value','P&L','P&L %'
            ])
            st.success("Portfolio cleared!")
            st.rerun()

    st.markdown("---")

    # Display portfolio if it exists
    if not st.session_state.portfolio.empty:
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(st.session_state.portfolio.copy())

        if metrics:
            # Portfolio summary
            st.markdown("### Portfolio Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card(
                    "Total Value",
                    f"${metrics['total_value']:,.2f}",
                    f"{metrics['num_positions']} positions"
                )

            with col2:
                pnl_type ="positive" if metrics['total_pnl'] >= 0 else"negative"
                metric_card(
                    "Total P&L",
                    f"${metrics['total_pnl']:,.2f}",
                    f"{metrics['total_pnl_pct']:+.2f}%",
                    pnl_type
                )

            with col3:
                metric_card(
                    "Diversification",
                    f"{metrics['diversification_score']:.1f}/100",
                    "Concentration Risk"
                )

            with col4:
                metric_card(
                    "Avg Volatility",
                    f"{metrics['avg_volatility']:.1f}%",
                    "Annualized"
                )

            st.markdown("---")

            # Portfolio risk metrics
            st.markdown("### Risk-Adjusted Performance Metrics")

            risk_metrics = calculate_sharpe_ratio(st.session_state.portfolio.copy())

            if risk_metrics:
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    metric_card(
                        "Annual Return",
                        f"{risk_metrics['annual_return']:.2%}",
                        "Expected",
                        "positive" if risk_metrics['annual_return'] > 0 else"negative"
                    )

                with col2:
                    metric_card(
                        "Annual Volatility",
                        f"{risk_metrics['annual_volatility']:.2%}",
                        "Risk"
                    )

                with col3:
                    sharpe_color ="positive" if risk_metrics['sharpe_ratio'] > 1 else"neutral"
                    metric_card(
                        "Sharpe Ratio",
                        f"{risk_metrics['sharpe_ratio']:.2f}",
                        "Risk-Adjusted",
                        sharpe_color
                    )

                with col4:
                    sortino_color ="positive" if risk_metrics['sortino_ratio'] > 1 else"neutral"
                    metric_card(
                        "Sortino Ratio",
                        f"{risk_metrics['sortino_ratio']:.2f}",
                        "Downside Risk",
                        sortino_color
                    )

                with col5:
                    metric_card(
                        "Max Drawdown",
                        f"{risk_metrics['max_drawdown']:.2%}",
                        "Worst Decline",
                        "negative"
                    )

            st.markdown("---")

            # Portfolio visualization
            col1, col2 = st.columns(2)

            with col1:
                # Allocation chart
                st.markdown("### Portfolio Allocation")
                allocation_fig = create_allocation_chart(st.session_state.portfolio.copy())
                if allocation_fig.data:
                    st.plotly_chart(allocation_fig, use_container_width=True)

            with col2:
                # P&L chart
                st.markdown("### Position Performance")

                portfolio_copy = st.session_state.portfolio.copy()
                portfolio_copy['Total Value'] = portfolio_copy['Shares'] * portfolio_copy['Current Price']
                portfolio_copy['Total Cost'] = portfolio_copy['Shares'] * portfolio_copy['Cost Basis']
                portfolio_copy['P&L'] = portfolio_copy['Total Value'] - portfolio_copy['Total Cost']

                fig = go.Figure(data=[
                    go.Bar(
                        x=portfolio_copy['Ticker'],
                        y=portfolio_copy['P&L'],
                        marker_color=['green' if pnl >= 0 else'red' for pnl in portfolio_copy['P&L']],
                        text=[f"${pnl:,.0f}" for pnl in portfolio_copy['P&L']],
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    title="P&L by Position",
                    yaxis_title="P&L ($)",
                    height=400,
                    template="plotly_white",
                    font=dict(family='Poppins, sans-serif'),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Advanced portfolio analytics
            if len(st.session_state.portfolio) >= 2:
                st.markdown("### Advanced Portfolio Analytics")

                col1, col2 = st.columns(2)

                with col1:
                    # Correlation heatmap
                    st.markdown("#### Correlation Matrix")
                    corr_fig = create_correlation_heatmap(st.session_state.portfolio.copy())
                    if corr_fig:
                        st.plotly_chart(corr_fig, use_container_width=True)
                    else:
                        st.info("Add more positions to view correlation matrix")

                with col2:
                    st.markdown("#### Portfolio Insights")
                    st.markdown("""
                    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; height: 480px;">
                        <h4 style="color: #f8fafc; margin-top: 0;">Understanding Your Metrics</h4>
                        <p style="color: #94a3b8; line-height: 1.6; font-size: 0.9rem;"><strong style="color: #f8fafc;">Sharpe Ratio:</strong> Measures risk-adjusted returns. Above 1 is good, above 2 is excellent.</p>
                        <p style="color: #94a3b8; line-height: 1.6; font-size: 0.9rem;"><strong style="color: #f8fafc;">Sortino Ratio:</strong> Similar to Sharpe but only considers downside volatility.</p>
                        <p style="color: #94a3b8; line-height: 1.6; font-size: 0.9rem;"><strong style="color: #f8fafc;">Max Drawdown:</strong> Largest peak-to-trough decline. Lower is better.</p>
                        <p style="color: #94a3b8; line-height: 1.6; font-size: 0.9rem;"><strong style="color: #f8fafc;">Correlation:</strong> Values close to 1 indicate assets move together. Diversify with low correlations.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Efficient frontier
                st.markdown("#### Efficient Frontier - Portfolio Optimization")
                frontier_fig = create_efficient_frontier(st.session_state.portfolio.copy())
                if frontier_fig:
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    st.info(" **Tip:** The green diamond shows the optimal risk-return portfolio. Your portfolio (red star) can be improved by moving towards the efficient frontier.")
                else:
                    st.info("Add more positions to view efficient frontier analysis")

            st.markdown("---")

            # ================================================================
            # PORTFOLIO REGIME SIMULATOR
            # ================================================================
            st.markdown("### Portfolio Regime Simulator")

            st.markdown("""
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                <p style="color: #1e40af; margin: 0; font-size: 0.9rem;">
                     <strong>How it works:</strong> This simulator analyzes how your portfolio historically performed during different volatility regimes
                    (based on VIX levels). Understanding regime-specific performance helps you build a more resilient allocation.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if len(st.session_state.portfolio) >= 2:
                # Prepare data for simulation
                portfolio_df = st.session_state.portfolio.copy()
                portfolio_df['Total Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
                total_value = portfolio_df['Total Value'].sum()
                portfolio_df['Weight'] = portfolio_df['Total Value'] / total_value

                tickers = tuple(portfolio_df['Ticker'].tolist())
                weights = tuple(portfolio_df['Weight'].tolist())

                with st.spinner("Analyzing portfolio performance across volatility regimes..."):
                    regime_results = calculate_regime_performance(tickers, weights)

                if regime_results:
                    # Display regime cards
                    st.markdown("#### Performance by Regime (Past 2 Years)")

                    regime_cols = st.columns(4)
                    for i, regime in enumerate(['Low','Medium','High','Extreme']):
                        with regime_cols[i]:
                            render_regime_card(regime, regime_results[regime])

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Comparison chart
                    comparison_chart = create_regime_comparison_chart(regime_results)
                    st.plotly_chart(comparison_chart, use_container_width=True)

                    # Insight box
                    insight = generate_regime_insight(regime_results)
                    st.markdown(f"""
                    <div style="background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 0 12px 12px 0; padding: 1.25rem; margin-top: 1rem;">
                        <div style="font-weight: 600; color: #0369a1; margin-bottom: 0.5rem;"> Portfolio Insight</div>
                        <div style="color: #0c4a6e; font-size: 0.95rem;">{insight}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional interpretation guide
                    with st.expander(" How to Interpret These Results"):
                        st.markdown("""
                        **Regime Definitions (based on VIX levels):**
                        - **Low Volatility** (VIX < 15): Calm, bullish markets with low uncertainty
                        - **Medium Volatility** (VIX 15-20): Normal market conditions with moderate swings
                        - **High Volatility** (VIX 20-30): Elevated uncertainty, often during corrections
                        - **Extreme Volatility** (VIX > 30): Crisis periods with major market stress

                        **What to Look For:**
                        - **Consistent returns across regimes** = Good diversification
                        - **Large negative returns in High/Extreme** = Consider adding defensive assets
                        - **High Sharpe in Low regime, negative in High** = Typical equity-heavy portfolio
                        - **Low max drawdown in Extreme** = Portfolio has good downside protection

                        **Actions to Consider:**
                        - Add bonds (TLT, AGG, BND) to reduce volatility sensitivity
                        - Add gold (GLD, IAU) for crisis hedging
                        - Consider VIX-based hedges for extreme scenarios
                        - Increase cash allocation if max drawdown is concerning
                        """)
                else:
                    st.warning("Unable to calculate regime performance. Some tickers may not have sufficient historical data.")
            else:
                st.info("Add at least 2 positions to your portfolio to run the regime simulator.")

            st.markdown("---")

            # Detailed portfolio table
            st.markdown("### Portfolio Details")

            # Prepare display dataframe
            display_df = st.session_state.portfolio.copy()
            display_df['Total Value'] = display_df['Shares'] * display_df['Current Price']
            display_df['Total Cost'] = display_df['Shares'] * display_df['Cost Basis']
            display_df['P&L'] = display_df['Total Value'] - display_df['Total Cost']
            display_df['P&L %'] = (display_df['P&L'] / display_df['Total Cost']) * 100
            display_df['Weight'] = display_df['Total Value'] / display_df['Total Value'].sum() * 100

            # Format for display
            display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.2f}")
            display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:.2f}")
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
            display_df['Total Value'] = display_df['Total Value'].apply(lambda x: f"${x:,.2f}")
            display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:,.2f}")
            display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:+.2f}%")
            display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(
                display_df[['Ticker','Shares','Cost Basis','Current Price','Total Value','P&L','P&L %','Weight']],
                use_container_width=True,
                hide_index=True
            )

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background-color: #1e293b; border: 1px solid #334155; border-radius: 1rem; margin: 2rem 0;">
            <h3 style="color: #f8fafc; margin-bottom: 1rem;">No Positions Yet</h3>
            <p style="color: #94a3b8; margin: 0;">Add your first position above to start tracking your portfolio performance and risk metrics.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ =="__main__":
    show()


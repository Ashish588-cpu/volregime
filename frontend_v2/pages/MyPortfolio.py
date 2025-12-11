"""
My Portfolio page - Track holdings, P&L, and price alerts
Displays portfolio with pie charts, performance metrics, and alert management
"""

import streamlit as st
import plotly.graph_objects as go
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.portfolio_manager import PortfolioManager, PriceAlertManager


def initialize_session_state():
    """Initialize portfolio and alerts in session state"""
    if 'portfolio_holdings' not in st.session_state:
        st.session_state.portfolio_holdings = []
    if 'price_alerts' not in st.session_state:
        st.session_state.price_alerts = []


def render_portfolio_summary(metrics: dict):
    """Display portfolio summary metrics"""
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Portfolio Summary</h2>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    pnl_color = "#00eaff" if metrics['total_pnl'] >= 0 else "#ff2b6b"
    pnl_arrow = "▲" if metrics['total_pnl'] >= 0 else "▼"

    with col1:
        st.markdown(f"""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        ">
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">
                Total Value
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; color: #00eaff; font-size: 1.75rem; font-weight: 700;">
                ${metrics['total_value']:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        ">
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">
                Total Cost
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; color: #f8fafc; font-size: 1.75rem; font-weight: 700;">
                ${metrics['total_cost']:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="
            background: #0d0d10;
            border: 1px solid {pnl_color};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        ">
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">
                Total P/L
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; color: {pnl_color}; font-size: 1.75rem; font-weight: 700;">
                {pnl_arrow} ${abs(metrics['total_pnl']):,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="
            background: #0d0d10;
            border: 1px solid {pnl_color};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        ">
            <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">
                Return %
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; color: {pnl_color}; font-size: 1.75rem; font-weight: 700;">
                {metrics['total_pnl_pct']:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_portfolio_charts(metrics: dict):
    """Render pie charts for portfolio allocation"""
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Portfolio Allocation</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Ticker allocation pie chart
    with col1:
        if metrics['ticker_allocation']:
            labels = list(metrics['ticker_allocation'].keys())
            values = list(metrics['ticker_allocation'].values())

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=['#00eaff', '#ff2b6b', '#00ff88', '#ffd700', '#9d4edd', '#06ffa5'],
                    line=dict(color='#0d0d10', width=2)
                ),
                textfont=dict(size=14, family='Inter', color='#f8fafc'),
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'By Ticker',
                    'font': {'size': 18, 'family': 'Space Grotesk', 'color': '#00eaff'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                paper_bgcolor='#0d0d10',
                plot_bgcolor='#0d0d10',
                font=dict(color='#f8fafc', family='Inter'),
                showlegend=True,
                legend=dict(
                    font=dict(size=12, family='Inter', color='#94a3b8'),
                    bgcolor='rgba(13,13,16,0.8)',
                    bordercolor='#2d3e5f',
                    borderwidth=1
                ),
                height=400,
                margin=dict(t=60, b=20, l=20, r=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    # Sector allocation pie chart
    with col2:
        if metrics['sector_allocation']:
            labels = list(metrics['sector_allocation'].keys())
            values = list(metrics['sector_allocation'].values())

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=['#00eaff', '#ff2b6b', '#00ff88', '#ffd700', '#9d4edd', '#06ffa5'],
                    line=dict(color='#0d0d10', width=2)
                ),
                textfont=dict(size=14, family='Inter', color='#f8fafc'),
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                title={
                    'text': 'By Sector',
                    'font': {'size': 18, 'family': 'Space Grotesk', 'color': '#00eaff'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                paper_bgcolor='#0d0d10',
                plot_bgcolor='#0d0d10',
                font=dict(color='#f8fafc', family='Inter'),
                showlegend=True,
                legend=dict(
                    font=dict(size=12, family='Inter', color='#94a3b8'),
                    bgcolor='rgba(13,13,16,0.8)',
                    bordercolor='#2d3e5f',
                    borderwidth=1
                ),
                height=400,
                margin=dict(t=60, b=20, l=20, r=20)
            )

            st.plotly_chart(fig, use_container_width=True)


def render_holdings_table(portfolio_manager: PortfolioManager):
    """Display holdings in a table format"""
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Current Holdings</h2>
    """, unsafe_allow_html=True)

    if not st.session_state.portfolio_holdings:
        st.info("No holdings yet. Add your first position below.")
        return

    for idx, holding in enumerate(st.session_state.portfolio_holdings):
        position = portfolio_manager.calculate_position_pnl(
            holding['ticker'],
            holding['quantity'],
            holding['purchase_price']
        )

        if position:
            pnl_color = "#00eaff" if position['pnl'] >= 0 else "#ff2b6b"
            pnl_arrow = "▲" if position['pnl'] >= 0 else "▼"

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"""
                <div style="
                    background: #0d0d10;
                    border: 1px solid #2d3e5f;
                    border-left: 4px solid {pnl_color};
                    border-radius: 12px;
                    padding: 1.25rem;
                    margin-bottom: 0.75rem;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <div>
                            <span style="font-family: 'Space Grotesk', sans-serif; color: #00eaff; font-size: 1.25rem; font-weight: 700;">
                                {position['ticker']}
                            </span>
                            <span style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.9rem; margin-left: 0.75rem;">
                                {position['name']}
                            </span>
                        </div>
                        <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.85rem;">
                            {position['sector']}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; font-family: 'Inter', sans-serif;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Quantity</div>
                            <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 600;">{position['quantity']:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Avg Cost</div>
                            <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 600;">${position['purchase_price']:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Current Price</div>
                            <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 600;">${position['current_price']:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">Total Value</div>
                            <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 600;">${position['current_value']:,.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.25rem;">P/L</div>
                            <div style="color: {pnl_color}; font-size: 0.95rem; font-weight: 700;">
                                {pnl_arrow} ${abs(position['pnl']):,.2f} ({position['pnl_pct']:+.2f}%)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                    st.session_state.portfolio_holdings.pop(idx)
                    st.rerun()


def render_add_holding_form():
    """Form to add new holdings"""
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Add New Position</h2>
    """, unsafe_allow_html=True)

    with st.form("add_holding_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL")
        with col2:
            quantity = st.number_input("Quantity", min_value=0.01, step=0.01, format="%.2f")
        with col3:
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.01, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Add Position", use_container_width=True, type="primary")

        if submitted:
            if ticker and quantity > 0 and purchase_price > 0:
                ticker = ticker.upper().strip()

                # Check if ticker already exists
                existing = [h for h in st.session_state.portfolio_holdings if h['ticker'] == ticker]
                if existing:
                    st.warning(f"{ticker} already exists in your portfolio. Remove it first to update.")
                else:
                    st.session_state.portfolio_holdings.append({
                        'ticker': ticker,
                        'quantity': quantity,
                        'purchase_price': purchase_price
                    })
                    st.success(f"Added {quantity} shares of {ticker} at ${purchase_price:.2f}")
                    st.rerun()
            else:
                st.error("Please fill in all fields with valid values")


def render_price_alerts(alert_manager: PriceAlertManager):
    """Display and manage price alerts"""
    st.markdown("---")
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Price Alerts</h2>
    """, unsafe_allow_html=True)

    # Display active alerts
    if st.session_state.price_alerts:
        st.markdown("""
        <h3 style="
            font-family: 'Space Grotesk', sans-serif;
            color: #94a3b8;
            font-size: 1.25rem;
            margin: 1.5rem 0 0.75rem 0;
        ">Active Alerts</h3>
        """, unsafe_allow_html=True)

        for idx, alert in enumerate(st.session_state.price_alerts):
            # Check alert status
            alert_status = alert_manager.check_alert(
                alert['ticker'],
                alert['target_price'],
                alert['alert_type']
            )

            if alert_status:
                status_color = "#00ff88" if alert_status['triggered'] else "#94a3b8"
                status_text = "TRIGGERED" if alert_status['triggered'] else "Active"
                arrow = "↑" if alert['alert_type'] == "above" else "↓"

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"""
                    <div style="
                        background: #0d0d10;
                        border: 1px solid {status_color};
                        border-radius: 12px;
                        padding: 1.25rem;
                        margin-bottom: 0.75rem;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-family: 'Space Grotesk', sans-serif; color: #00eaff; font-size: 1.15rem; font-weight: 700;">
                                    {alert['ticker']}
                                </span>
                                <span style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.9rem; margin-left: 0.75rem;">
                                    Alert when price goes {alert['alert_type']} ${alert['target_price']:.2f} {arrow}
                                </span>
                            </div>
                            <div style="display: flex; gap: 1.5rem; align-items: center;">
                                <div>
                                    <div style="font-family: 'Inter', sans-serif; color: #94a3b8; font-size: 0.75rem;">Current Price</div>
                                    <div style="font-family: 'Space Grotesk', sans-serif; color: #f8fafc; font-size: 1rem; font-weight: 600;">
                                        ${alert_status['current_price']:.2f}
                                    </div>
                                </div>
                                <div style="
                                    padding: 4px 12px;
                                    background: {status_color}22;
                                    border: 1px solid {status_color};
                                    border-radius: 6px;
                                    color: {status_color};
                                    font-family: 'Inter', sans-serif;
                                    font-size: 0.75rem;
                                    font-weight: 600;
                                ">{status_text}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("Delete", key=f"delete_alert_{idx}", use_container_width=True):
                        st.session_state.price_alerts.pop(idx)
                        st.rerun()
    else:
        st.info("No active alerts. Create your first alert below.")

    # Add new alert form
    st.markdown("""
    <h3 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #94a3b8;
        font-size: 1.25rem;
        margin: 2rem 0 0.75rem 0;
    ">Create New Alert</h3>
    """, unsafe_allow_html=True)

    with st.form("add_alert_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            alert_ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL", key="alert_ticker")
        with col2:
            alert_type = st.selectbox("Alert Type", ["above", "below"])
        with col3:
            target_price = st.number_input("Target Price ($)", min_value=0.01, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Create Alert", use_container_width=True, type="primary")

        if submitted:
            if alert_ticker and target_price > 0:
                alert_ticker = alert_ticker.upper().strip()

                # Verify ticker exists
                stock_info = alert_manager.get_stock_info(alert_ticker)
                if stock_info:
                    st.session_state.price_alerts.append({
                        'ticker': alert_ticker,
                        'target_price': target_price,
                        'alert_type': alert_type
                    })
                    st.success(f"Alert created for {alert_ticker} when price goes {alert_type} ${target_price:.2f}")
                    st.rerun()
                else:
                    st.error(f"Invalid ticker symbol: {alert_ticker}")
            else:
                st.error("Please fill in all fields with valid values")


def show():
    """Main portfolio page"""
    initialize_session_state()

    # Check authentication
    from utils.auth import is_authenticated

    # Hero header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d0d10, #1a1a1d);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #00eaff;
        box-shadow: 0 0 30px rgba(0,234,255,0.3);
    ">
        <h1 style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00eaff, #ff2b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 1rem 0;
        ">My Portfolio</h1>
        <p style="
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0;">
            Track your holdings, P/L, and price alerts
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Require authentication
    if not is_authenticated():
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: #0b1220;
                border: 2px solid #00e6f6;
                border-radius: 16px;
                padding: 3rem;
                text-align: center;
                box-shadow: 0 0 30px rgba(0,230,246,0.3);
            ">
                <h2 style="color: #00e6f6; margin-bottom: 1rem; font-size: 1.75rem;">
                    Login Required
                </h2>
                <p style="color: #e2e8f0; line-height: 1.6; margin-bottom: 2rem;">
                    Sign in to access your portfolio tracking, manage holdings, and set price alerts.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Login / Sign Up", key="portfolio_login_btn", use_container_width=True, type="primary"):
                    st.session_state.current_page = "Auth"
                    st.rerun()
            with col_b:
                if st.button("← Back to Home", key="portfolio_back_home", use_container_width=True):
                    st.session_state.current_page = "Home"
                    st.rerun()

        return

    # Get user email from session
    user_email = st.session_state.get('user', {}).get('email', 'demo@user.com')

    portfolio_manager = PortfolioManager(user_email)
    alert_manager = PriceAlertManager(user_email)

    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="portfolio_back_to_home"):
            st.session_state.current_page = "Home"
            st.rerun()

    # Calculate portfolio metrics
    if st.session_state.portfolio_holdings:
        metrics = portfolio_manager.calculate_portfolio_metrics(st.session_state.portfolio_holdings)

        # Portfolio summary
        render_portfolio_summary(metrics)

        # Portfolio charts
        render_portfolio_charts(metrics)

        # Holdings table
        render_holdings_table(portfolio_manager)

    # Add holding form
    render_add_holding_form()

    # Price alerts
    render_price_alerts(alert_manager)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem 0;
    ">
        <p>Portfolio data is stored in your browser session. Real-time prices from Yahoo Finance.</p>
        <p style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #1e293b;">2025 || Ashish Dahal || copy right reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="VolRegime - My Portfolio",
        layout="wide"
    )
    show()

"""
User Guide
Simple documentation for using the platform
"""

import streamlit as st

def show():
    """Shows user guide page"""

    # Back button
    if st.button("← Back to Home", key="guide_back_btn"):
        st.session_state.current_page = "Home"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Center all content
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("""
        <h1 style="
            color: #00e6f6;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
        ">User Guide</h1>
        """, unsafe_allow_html=True)

        # Video Tutorial Section
        st.markdown("""
        <h2 style="
            color: #00e6f6;
            font-size: 1.75rem;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        ">Video Tutorial</h2>
        """, unsafe_allow_html=True)

        # Video embed from YouTube
        try:
            st.video("https://youtu.be/Nhykacq8LWA")
        except Exception as e:
            st.warning(f"Unable to load video: {e}")

        st.markdown("---")

        # Getting Started
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Getting Started</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p>VolRegime is a financial intelligence platform for market analysis and portfolio tracking.</p>
        <ul>
            <li>View live market data</li>
            <li>Track your portfolio</li>
            <li>Use advanced analytics tools</li>
            <li>Monitor market indicators</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Market Dashboard
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Market Dashboard</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p>The home page shows live market snapshots.</p>
        <ul>
            <li>S&P 500 index</li>
            <li>NASDAQ index</li>
            <li>DOW index</li>
            <li>VIX volatility</li>
        </ul>
        <p>Data refreshes every 10 seconds automatically.</p>
        <p>Click any index card to view detailed charts.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Advanced Tools
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Advanced Tools</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p>Portfolio Tools page provides advanced analytics.</p>
        <ul>
            <li>Efficient frontier calculation</li>
            <li>Risk metrics (Sharpe ratio, volatility)</li>
            <li>Correlation matrix</li>
            <li>Portfolio optimization</li>
        </ul>
        <p>Enter your tickers and allocation weights.</p>
        <p>Use dropdowns to select indicators and timeframes.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Chart Indicators
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Chart Indicators Explained</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p><strong>RSI (Relative Strength Index)</strong><br>
        Measures momentum from 0 to 100.<br>
        Above 70 = overbought. Below 30 = oversold.</p>

        <p><strong>Bollinger Bands</strong><br>
        Shows price volatility bands.<br>
        Price touching upper band = high volatility.<br>
        Price touching lower band = low volatility.</p>

        <p><strong>SMA (Simple Moving Average)</strong><br>
        Average price over N days.<br>
        Smooths out price trends.</p>

        <p><strong>MACD</strong><br>
        Trend following momentum indicator.<br>
        Signal line crossovers indicate buy/sell points.</p>

        <p><strong>Sharpe Ratio</strong><br>
        Risk adjusted return measure.<br>
        Higher is better. Above 1.0 is good.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Portfolio Tracker
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Portfolio Tracker</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p>My Portfolio page lets you track holdings.</p>
        <ul>
            <li>Add tickers and share counts</li>
            <li>Enter your cost basis</li>
            <li>View total value and P&L</li>
            <li>See allocation breakdown</li>
        </ul>
        <p>Data updates when you reload the page.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Login & Account
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Login & Account</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p>Click Login button at top right.</p>
        <ul>
            <li>Create account with email</li>
            <li>Sign in to save portfolio</li>
            <li>Reset password if needed</li>
            <li>Or continue as guest</li>
        </ul>
        <p>Guest mode works for all features except saving data.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Troubleshooting
        st.markdown("""
        <h2 style="color: #00e6f6; font-size: 1.75rem; margin-top: 2rem;">Troubleshooting</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color: #e2e8f0; line-height: 1.8;">
        <p><strong>Charts not loading:</strong><br>
        Refresh the page. Check internet connection.</p>

        <p><strong>Data looks wrong:</strong><br>
        Wait for next auto refresh (10 seconds).<br>
        Market data comes from Yahoo Finance.</p>

        <p><strong>Dropdown not visible:</strong><br>
        Click the dropdown again.<br>
        Options have dark theme styling.</p>

        <p><strong>Portfolio not saving:</strong><br>
        You must be logged in to save data.<br>
        Guest mode does not persist data.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

    # Copyright footer
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


if __name__ == "__main__":
    st.set_page_config(
        page_title="VolRegime - User Guide",
        layout="wide"
    )
    show()

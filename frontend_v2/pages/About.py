"""
About page - Information about VolRegime platform
"""

import streamlit as st


def show():
    """shows the about page with platform information"""

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
        ">About VolRegime</h1>
        <p style="
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0;">
            Your comprehensive financial intelligence platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mission statement
    st.markdown("""
    <div style="
        background: #0d0d10;
        border-left: 4px solid #00eaff;
        border-radius: 0 12px 12px 0;
        padding: 2rem;
        margin-bottom: 2rem;
    ">
        <h2 style="
            font-family: 'Space Grotesk', sans-serif;
            color: #00eaff;
            font-size: 1.75rem;
            margin: 0 0 1rem 0;
        ">Our Mission</h2>
        <p style="
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            font-size: 1rem;
            line-height: 1.8;
            margin: 0;
        ">
            VolRegime is designed to empower traders and investors with advanced market analytics,
            real-time data, and intelligent volatility regime detection. We believe that sophisticated
            financial tools should be accessible to everyone, not just institutional investors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Features section
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Platform Features</h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #00eaff;
                margin-bottom: 1rem;
            ">Real-Time Market Data</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                Live price feeds, technical indicators, and market snapshots updated every 10 seconds
                from major exchanges and indices.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #ff2b6b;
                margin-bottom: 1rem;
            ">Advanced Analytics</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                Comprehensive technical analysis tools including RSI, MACD, moving averages, volatility
                cones, and drawdown analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #00eaff;
                margin-bottom: 1rem;
            ">Volatility Regime Detection</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                AI-powered market state identification to help you adapt your strategy to current
                market conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #ff2b6b;
                margin-bottom: 1rem;
            ">Portfolio Management</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                Build and track custom portfolios with optimization tools and risk-adjusted
                performance metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #00eaff;
                margin-bottom: 1rem;
            ">Market News & Learning</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                Curated news from top financial sources and structured learning paths from beginner
                to advanced trader.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #ff2b6b;
                margin-bottom: 1rem;
            ">Customizable Dashboards</h3>
            <p style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.6;
            ">
                Personalize your experience with watchlists, custom timeframes, and saved preferences
                across all features.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("---")
    st.markdown("""
    <h2 style="
        font-family: 'Space Grotesk', sans-serif;
        color: #00eaff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        text-shadow: 0 0 20px rgba(0,234,255,0.5);
    ">Technology Stack</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #00eaff;
                font-size: 1.25rem;
                margin-bottom: 1rem;
            ">Data Sources</h3>
            <ul style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.8;
            ">
                <li>Yahoo Finance API for real-time market data</li>
                <li>Multiple RSS feeds from MarketWatch, Reuters, WSJ, CNBC, Bloomberg</li>
                <li>Technical indicators calculated using industry-standard algorithms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: #0d0d10;
            border: 1px solid #2d3e5f;
            border-radius: 12px;
            padding: 1.5rem;
        ">
            <h3 style="
                font-family: 'Space Grotesk', sans-serif;
                color: #ff2b6b;
                font-size: 1.25rem;
                margin-bottom: 1rem;
            ">Built With</h3>
            <ul style="
                font-family: 'Inter', sans-serif;
                color: #94a3b8;
                font-size: 0.9rem;
                line-height: 1.8;
            ">
                <li>Streamlit for interactive web interface</li>
                <li>Python for data processing and analytics</li>
                <li>Plotly for professional charting and visualizations</li>
                <li>Supabase for secure user authentication</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="
        background: rgba(255, 43, 107, 0.1);
        border: 1px solid #ff2b6b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    ">
        <h3 style="
            font-family: 'Space Grotesk', sans-serif;
            color: #ff2b6b;
            font-size: 1.25rem;
            margin: 0 0 1rem 0;
        ">Important Disclaimer</h3>
        <p style="
            font-family: 'Inter', sans-serif;
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 0;
        ">
            VolRegime is an educational and analytical tool. The information provided on this platform
            is for informational purposes only and should not be considered financial advice. Always
            do your own research and consult with a qualified financial advisor before making investment
            decisions. Past performance does not guarantee future results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Back to home button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Back to Home", use_container_width=True, type="primary"):
            st.session_state.current_page = "Home"
            st.rerun()

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


if __name__ == "__main__":
    st.set_page_config(
        page_title="VolRegime - About",
        layout="wide"
    )
    show()

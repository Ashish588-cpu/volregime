"""
Home page - shows live market data for major assets
Pretty simple, just fetches prices from Yahoo Finance and displays them in cards
"""

import streamlit as st
import yfinance as yf
from datetime import datetime
import sys
import os

# need parent dir in path to import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import news fetching function from ArticlesLearn page
try:
    import feedparser
    ARTICLES_AVAILABLE = True
except ImportError:
    ARTICLES_AVAILABLE = False

# try to use auto-refresh if it's installed (not required though)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

# refresh page every 10 seconds to get new prices
if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=10000, key="home_refresh")


# Data fetching functions

@st.cache_data(ttl=10, show_spinner=False)  # cache for 10 seconds
def get_asset_data(ticker: str, period: str = "5d") -> dict:
    """grabs price data from yahoo finance"""
    try:
        # get data for specified period
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period)

        # need at least 2 data points to calculate change
        if data.empty or len(data) < 2:
            return None

        # current price vs previous close
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        # get the full name of the asset
        info = ticker_obj.info
        name = info.get('shortName', ticker)

        return {
            'ticker': ticker,
            'name': name,
            'price': current_price,
            'change': change,
            'change_pct': change_pct
        }
    except Exception:
        return None  # if something breaks just return nothing


@st.cache_data(ttl=10, show_spinner=False)
def get_market_snapshot(period: str = "5d") -> dict:
    """gets data for all the assets we want to show"""
    # the assets we're tracking
    assets = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'DOW',
        '^VIX': 'VIX'
    }

    snapshot = {}
    for ticker, display_name in assets.items():
        data = get_asset_data(ticker, period)
        if data:
            data['display_name'] = display_name
            snapshot[ticker] = data

    return snapshot


@st.cache_data(ttl=300, show_spinner=False)
def get_featured_articles(limit: int = 3) -> list:
    """Fetch a few featured articles for home page preview"""
    if not ARTICLES_AVAILABLE:
        return []

    try:
        import feedparser
        news_sources = [
            {'name': 'MarketWatch', 'url': 'https://feeds.marketwatch.com/marketwatch/topstories/'},
            {'name': 'Yahoo Finance', 'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline'},
        ]

        articles = []
        for source in news_sources:
            try:
                feed = feedparser.parse(source['url'])
                for entry in feed.entries[:3]:  # Get 3 from each source
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')

                    if title and link:
                        # Clean up summary
                        if not summary:
                            summary = "Click to read more..."

                        articles.append({
                            'title': title,
                            'summary': summary[:120] + '...' if len(summary) > 120 else summary,
                            'link': link,
                            'source': source['name']
                        })

                        if len(articles) >= limit:
                            return articles[:limit]
            except Exception as e:
                # Log error but continue to next source
                continue

        return articles[:limit]
    except Exception as e:
        return []


# UI stuff - displaying the data

def render_asset_card(ticker: str, data: dict):
    """makes a card showing the asset price and change"""
    # single neon cyan accent only
    arrow = "▲" if data['change_pct'] >= 0 else "▼"

    # format price nicely
    price = float(data['price'])
    if price > 1000:
        price_str = f"${price:,.2f}"
    elif price > 100:
        price_str = f"${price:.2f}"
    else:
        price_str = f"{price:.2f}"

    # format change
    change_val = float(data['change'])
    change_pct = float(data['change_pct'])

    card_html = f"""
    <div style="
        background: #0b1220;
        border: 1px solid #00e6f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    " onmouseover="this.style.boxShadow='0 0 15px rgba(0,230,246,0.3)'" onmouseout="this.style.boxShadow='none'">
        <div style="color: #8b9db8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.5rem;">
            {ticker} • {data['display_name']}
        </div>
        <div style="color: #e2e8f0; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
            {price_str}
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #00e6f6; font-size: 1rem; font-weight: 600;">
                {arrow} {abs(change_val):.2f}
            </span>
            <span style="color: #00e6f6; font-size: 0.9rem; font-weight: 600; background: rgba(0,230,246,0.1); padding: 0.25rem 0.75rem; border-radius: 6px;">
                {change_pct:+.2f}%
            </span>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    # button to go to detailed view
    if st.button(f"View {data['display_name']} Details", key=f"btn_{ticker}", use_container_width=True):
        st.session_state.selected_ticker = ticker
        st.session_state.current_page = "Asset Dashboard"
        st.rerun()


# Main page rendering

def show():
    """the actual home page that gets displayed"""

    # Onboarding popup (only on first visit)
    if 'hide_guide_popup' not in st.session_state:
        st.session_state.hide_guide_popup = False

    if not st.session_state.hide_guide_popup:
        st.markdown("""
        <style>
        .onboarding-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .onboarding-modal {
            background: #0b1220;
            border: 2px solid #00e6f6;
            border-radius: 16px;
            padding: 2.5rem;
            max-width: 500px;
            box-shadow: 0 0 40px rgba(0,230,246,0.4);
            animation: slideUp 0.4s ease;
        }
        @keyframes slideUp {
            from { transform: translateY(30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: #0b1220;
                border: 2px solid #00e6f6;
                border-radius: 16px;
                padding: 2.5rem;
                box-shadow: 0 0 40px rgba(0,230,246,0.4);
                text-align: center;
            ">
                <h2 style="color: #00e6f6; margin-bottom: 1rem; font-size: 1.75rem;">
                    Welcome — Need Help Navigating?
                </h2>
                <p style="color: #e2e8f0; line-height: 1.6; margin-bottom: 2rem;">
                    View the quick guide for using charts, portfolio tracking, and advanced tools.
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Open User Guide", key="open_guide_btn", use_container_width=True, type="primary"):
                    st.session_state.hide_guide_popup = True
                    st.session_state.current_page = "Guide"
                    st.rerun()
            with col_b:
                if st.button("Skip", key="skip_guide_btn", use_container_width=True):
                    st.session_state.hide_guide_popup = True
                    st.rerun()

    # Hero header
    st.markdown("""
    <div style="
        background: #0b1220;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #00e6f6;
        box-shadow: 0 0 25px rgba(0,230,246,0.2);
    ">
        <h1 style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: #00e6f6;
            margin: 0 0 1rem 0;
        ">VolRegime</h1>
        <p style="font-family: 'Inter', sans-serif; color: #8b9db8; font-size: 1.2rem; margin: 0;">
            Financial Intelligence Platform
        </p>
        <p style="font-family: 'Inter', sans-serif; color: #6b7c92; font-size: 0.9rem; margin-top: 0.5rem;">
            Realtime market data • Volatility analysis • Portfolio management
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Moving ticker bar with major indexes
    ticker_data = get_market_snapshot("1d")
    if ticker_data:
        ticker_items = []
        for _, data in ticker_data.items():
            price = data['price']
            change_pct = data['change_pct']
            display_name = data['display_name']
            arrow = "▲" if change_pct >= 0 else "▼"
            ticker_items.append(f'<span style="margin: 0 2rem; white-space: nowrap;"><strong>{display_name}:</strong> ${price:,.2f} <span style="color: #00e6f6;">{arrow} {abs(change_pct):.2f}%</span></span>')

        ticker_html = ''.join(ticker_items * 3)

        st.markdown(f"""
        <style>
        @keyframes scroll-ticker {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-33.33%); }}
        }}
        .ticker-container {{
            overflow: hidden;
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 10px;
            padding: 0.75rem 0;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 12px rgba(0,230,246,0.15);
        }}
        .ticker-content {{
            display: inline-flex;
            white-space: nowrap;
            animation: scroll-ticker 30s linear infinite;
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
            font-size: 0.9rem;
        }}
        </style>
        <div class="ticker-container">
            <div class="ticker-content">
                {ticker_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Live ticker
    now = datetime.now()
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 0.75rem;
        background: #0b1220;
        border-radius: 10px;
        border: 1px solid #00e6f6;
        box-shadow: 0 0 12px rgba(0,230,246,0.15);
        margin-bottom: 1.5rem;
    ">
        <span style="color: #00e6f6; font-weight: 600; letter-spacing: 1px;">
            LIVE DATA • Updated: {now.strftime("%I:%M:%S %p")}
        </span>
        <span style="color: #6b7c92; font-size: 0.85rem; margin-left: 1rem;">
            Auto refreshes every 10 seconds
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Section header
    st.markdown("""
    <h2 style="
        color: #00e6f6;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 1.5rem 0 0.75rem 0;
        text-align: center;
    ">Live Market Snapshot</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color: #8b9db8; text-align: center; margin-bottom: 1.5rem;">
        Click any tile to view detailed analysis with timeframe options
    </p>
    """, unsafe_allow_html=True)

    # Fetch data (always 5-day period for home page)
    with st.spinner("Loading live market data..."):
        market_data = get_market_snapshot("5d")

    if not market_data:
        st.error("Unable to load market data. Please check your internet connection.")
        return

    # Row 1: S&P 500, NASDAQ
    col1, col2 = st.columns(2)
    with col1:
        if '^GSPC' in market_data:
            render_asset_card('^GSPC', market_data['^GSPC'])
    with col2:
        if '^IXIC' in market_data:
            render_asset_card('^IXIC', market_data['^IXIC'])

    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    # Row 2: DOW, VIX
    col1, col2 = st.columns(2)
    with col1:
        if '^DJI' in market_data:
            render_asset_card('^DJI', market_data['^DJI'])
    with col2:
        if '^VIX' in market_data:
            render_asset_card('^VIX', market_data['^VIX'])

    # Article Preview Section
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
    ">Latest Market News</h2>
    """, unsafe_allow_html=True)

    featured_articles = get_featured_articles(3)
    if featured_articles and len(featured_articles) > 0:
        cols = st.columns(len(featured_articles))
        for idx, article in enumerate(featured_articles):
            with cols[idx]:
                st.markdown(f"""
                <div style="
                    background: #0d0d10;
                    border: 1px solid #2d3e5f;
                    border-radius: 12px;
                    padding: 1.25rem;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    transition: all 0.3s ease;
                ">
                    <div style="
                        font-family: 'Inter', sans-serif;
                        font-size: 0.75rem;
                        color: #00eaff;
                        font-weight: 600;
                        margin-bottom: 0.5rem;
                    ">{article['source']}</div>
                    <h4 style="
                        font-family: 'Inter', sans-serif;
                        color: #f8fafc;
                        font-size: 0.95rem;
                        font-weight: 600;
                        margin: 0 0 0.5rem 0;
                        line-height: 1.4;
                        flex-grow: 1;
                    ">
                        <a href="{article['link']}" target="_blank" style="text-decoration: none; color: inherit;">
                            {article['title'][:60]}{'...' if len(article['title']) > 60 else ''}
                        </a>
                    </h4>
                    <p style="
                        font-family: 'Inter', sans-serif;
                        color: #94a3b8;
                        font-size: 0.8rem;
                        margin: 0;
                        line-height: 1.4;
                    ">{article['summary'][:80]}{'...' if len(article['summary']) > 80 else ''}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("View All Articles", use_container_width=True, type="primary"):
                st.session_state.current_page = "Articles / Learn"
                st.rerun()
    else:
        st.info("News articles are loading... Please refresh the page if this persists.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'Inter', sans-serif; text-align: center; color: #64748b; font-size: 0.85rem; padding: 2rem 0;">
        <p>Data provided by Yahoo Finance • Auto-refreshes every 10 seconds</p>
        <p>Use the navigation bar above to explore more features</p>
    </div>
    """, unsafe_allow_html=True)


# Standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="VolRegime - Home",
        page_icon="📊",
        layout="wide"
    )

    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = "SPY"
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    show()

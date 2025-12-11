"""
Articles & Learning Hub - Financial Education and Market News
Comprehensive resource with categorized news, bookmarks, and structured learning paths
"""

import streamlit as st
import feedparser
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NEWS CATEGORY DEFINITIONS
# ============================================================================

NEWS_CATEGORIES = {
    "Market Updates": ["market","index","trading","stocks","dow","s&p","nasdaq","rally","selloff"],
    "Fed Policy": ["fed","federal reserve","interest rate","powell","inflation","monetary","fomc","rate hike","rate cut"],
    "Earnings": ["earnings","revenue","profit","quarterly","q1","q2","q3","q4","eps","guidance","beat","miss"],
    "Crypto": ["bitcoin","crypto","ethereum","blockchain","btc","eth","coinbase","binance","defi"],
    "Tech": ["apple","microsoft","google","amazon","meta","nvidia","tech","ai","artificial intelligence","semiconductor"]
}

CATEGORY_COLORS = {
    "All":"#6b7280",
    "Market Updates":"#3b82f6",
    "Fed Policy":"#8b5cf6",
    "Earnings":"#10b981",
    "Crypto":"#f59e0b",
    "Tech":"#06b6d4"
}


# ============================================================================
# LEARNING PATH DEFINITIONS
# ============================================================================

LEARNING_PATHS = {
    "beginner": {
        "title":" Beginner Track - Build Your Foundation",
        "description":"Start here if you're new to trading and volatility",
        "color":"#059669",
        "lessons": [
            {
                "title":"What is Volatility?",
                "description":"Understand what market volatility means, why it matters, and how it's measured using metrics like standard deviation and the VIX index.",
                "url":"https://www.investopedia.com/terms/v/volatility.asp"
            },
            {
                "title":"Understanding Market Indices",
                "description":"Learn about major indices like S&P 500, NASDAQ, and Dow Jones - how they're calculated and what they tell us about the market.",
                "url":"https://www.investopedia.com/terms/m/marketindex.asp"
            },
            {
                "title":"Risk vs Return Basics",
                "description":"Explore the fundamental relationship between risk and return, and why higher potential returns typically come with higher risk.",
                "url":"https://www.investopedia.com/terms/r/riskreturntradeoff.asp"
            },
            {
                "title":"How to Read Stock Charts",
                "description":"Master the basics of reading price charts, understanding candlesticks, and identifying basic patterns in market data.",
                "url":"https://www.investopedia.com/trading/candlestick-charting-what-is-it/"
            },
            {
                "title":"Building Your First Portfolio",
                "description":"Learn portfolio construction basics, asset allocation principles, and how to start investing with a diversified approach.",
                "url":"https://www.investopedia.com/terms/p/portfolio.asp"
            }
        ]
    },
    "intermediate": {
        "title":" Intermediate Track - Technical Analysis",
        "description":"Deepen your knowledge with technical indicators and analysis techniques",
        "color":"#f59e0b",
        "lessons": [
            {
                "title":"Moving Averages & Trends",
                "description":"Master simple and exponential moving averages, learn to identify trends, and understand crossover signals.",
                "url":"https://www.investopedia.com/terms/m/movingaverage.asp"
            },
            {
                "title":"RSI & MACD Indicators",
                "description":"Deep dive into momentum indicators - how RSI identifies overbought/oversold conditions and MACD signals trend changes.",
                "url":"https://www.investopedia.com/terms/r/rsi.asp"
            },
            {
                "title":"Volatility Regimes Explained",
                "description":"Understand how markets cycle through different volatility regimes and how to adapt your strategy accordingly.",
                "url":"https://www.investopedia.com/terms/v/vix.asp"
            },
            {
                "title":"Options Basics",
                "description":"Introduction to options trading - calls, puts, strike prices, expiration, and basic options strategies.",
                "url":"https://www.investopedia.com/options-basics-tutorial-4583012"
            },
            {
                "title":"Portfolio Diversification Strategies",
                "description":"Advanced diversification across asset classes, sectors, and geographies to optimize risk-adjusted returns.",
                "url":"https://www.investopedia.com/terms/d/diversification.asp"
            }
        ]
    },
    "advanced": {
        "title":" Advanced Track - Professional Strategies",
        "description":"Master institutional-grade analytics and quantitative methods",
        "color":"#dc2626",
        "lessons": [
            {
                "title":"Greeks & Options Pricing",
                "description":"Master Delta, Gamma, Theta, Vega, and Rho - understand how options are priced and how to manage Greek exposures.",
                "url":"https://www.investopedia.com/terms/g/greeks.asp"
            },
            {
                "title":"Volatility Surface Analysis",
                "description":"Analyze implied volatility surfaces, term structure, and volatility skew to identify trading opportunities.",
                "url":"https://www.investopedia.com/terms/v/volatility-smile.asp"
            },
            {
                "title":"Portfolio Optimization Algorithms",
                "description":"Learn Modern Portfolio Theory, mean-variance optimization, and efficient frontier construction techniques.",
                "url":"https://www.investopedia.com/terms/m/modernportfoliotheory.asp"
            },
            {
                "title":"Risk-Adjusted Performance Metrics",
                "description":"Master Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and other professional performance measures.",
                "url":"https://www.investopedia.com/terms/s/sharperatio.asp"
            },
            {
                "title":"Quantitative Trading Strategies",
                "description":"Explore systematic trading approaches, factor investing, statistical arbitrage, and algorithmic strategy development.",
                "url":"https://www.investopedia.com/terms/q/quantitative-trading.asp"
            }
        ]
    }
}

# ============================================================================
# NEWS FETCHING AND CATEGORIZATION
# ============================================================================

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_market_news():
    """Fetch latest market news from multiple RSS feeds"""
    try:
        news_sources = [
            {'name': 'MarketWatch', 'url': 'https://feeds.marketwatch.com/marketwatch/topstories/', 'icon': ''},
            {'name': 'Yahoo Finance', 'url': 'https://feeds.finance.yahoo.com/rss/2.0/headline', 'icon': ''},
            {'name': 'Reuters Business', 'url': 'https://feeds.reuters.com/reuters/businessNews', 'icon': ''},
            {'name': 'WSJ Markets', 'url': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml', 'icon': ''},
            {'name': 'CNBC', 'url': 'https://www.cnbc.com/id/100003114/device/rss/rss.html', 'icon': ''},
            {'name': 'Bloomberg', 'url': 'https://feeds.bloomberg.com/markets/news.rss', 'icon': ''},
            {'name': 'Seeking Alpha', 'url': 'https://seekingalpha.com/market_currents.xml', 'icon': ''}
        ]

        all_articles = []

        for source in news_sources:
            try:
                feed = feedparser.parse(source['url'])
                for entry in feed.entries[:6]: # Get top 6 from each source
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))

                    # Categorize article
                    category = categorize_article(title + " " + summary)

                    # Parse published date
                    published = entry.get('published', '')
                    try:
                        if published:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                        else:
                            pub_date = datetime.now()
                    except:
                        pub_date = datetime.now()

                    article = {
                        'title': title,
                        'summary': summary[:150] + '...' if len(summary) > 150 else summary,
                        'full_summary': summary,
                        'link': entry.get('link', ''),
                        'published': published,
                        'pub_date': pub_date,
                        'source': source['name'],
                        'icon': source['icon'],
                        'category': category
                    }
                    all_articles.append(article)
            except Exception:
                continue

        # Sort by publication date (newest first)
        all_articles.sort(key=lambda x: x['pub_date'], reverse=True)

        return all_articles

    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []


def categorize_article(text: str) -> str:
    """Categorize article based on keyword matching"""
    text_lower = text.lower()

    for category, keywords in NEWS_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category

    return "Market Updates" # Default category


def extract_trending_topics(articles: list) -> list:
    """Extract trending topics from article titles and summaries"""
    topic_counts = {}

    # Keywords to look for as trending topics
    trending_keywords = [
        "AI","Inflation","Rates","Earnings","Fed","Bitcoin","Crypto",
        "Tech","Oil","Gold","China","Jobs","GDP","Recession","Rally",
        "NVIDIA","Apple","Tesla","Microsoft","Amazon","Meta"
    ]

    for article in articles:
        text = article['title'] +"" + article.get('full_summary','')
        for keyword in trending_keywords:
            if keyword.lower() in text.lower():
                topic_counts[keyword] = topic_counts.get(keyword, 0) + 1

    # Sort by count and return top 10
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, count in sorted_topics[:10] if count >= 1]


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def render_news_card(article: dict, show_bookmark: bool = True, index: int = 0):
    """Render a single news article card"""
    import time
    category_color = CATEGORY_COLORS.get(article['category'],'#6b7280')
    # Use index and timestamp to ensure unique keys
    article_id = f"{index}_{abs(hash(article['link']))}_{int(time.time() * 1000) % 10000}"

    # Check if bookmarked
    if'bookmarks' not in st.session_state:
        st.session_state['bookmarks'] = []

    is_bookmarked = any(b['link'] == article['link'] for b in st.session_state['bookmarks'])
    bookmark_icon ="" if is_bookmarked else"️"

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; transition: all 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span style="font-size: 1.1rem;">{article['icon']}</span>
                <span style="background: #f1f5f9; color: #475569; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500;">{article['source']}</span>
                <span style="background: {category_color}20; color: {category_color}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{article['category']}</span>
            </div>
            <span style="color: #94a3b8; font-size: 0.8rem;">{article['published'][:16] if article['published'] else''}</span>
        </div>
        <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-weight: 600; font-size: 1rem; line-height: 1.4;">
            <a href="{article['link']}" target="_blank" style="text-decoration: none; color: inherit;">{article['title']}</a>
        </h4>
        <p style="color: #64748b; margin: 0; font-size: 0.875rem; line-height: 1.5;">{article['summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Bookmark button
    if show_bookmark:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button(bookmark_icon, key=f"bm_{article_id}", help="Bookmark this article"):
                toggle_bookmark(article)
                st.rerun()


def toggle_bookmark(article: dict):
    """Toggle bookmark status for an article"""
    if'bookmarks' not in st.session_state:
        st.session_state['bookmarks'] = []

    # Check if already bookmarked
    existing = [i for i, b in enumerate(st.session_state['bookmarks']) if b['link'] == article['link']]

    if existing:
        st.session_state['bookmarks'].pop(existing[0])
    else:
        st.session_state['bookmarks'].append({
            'title': article['title'],
            'link': article['link'],
            'source': article['source'],
            'category': article['category']
        })


def render_learning_lesson(lesson: dict, track_key: str, lesson_idx: int, color: str):
    """Render a single lesson card with progress tracking"""
    # Initialize progress tracking
    progress_key = f"completed_{track_key}"
    if progress_key not in st.session_state:
        st.session_state[progress_key] = []

    is_completed = lesson_idx in st.session_state[progress_key]
    check_icon ="" if is_completed else"⬜"

    st.markdown(f"""
    <div style="background: white; border-left: 4px solid {color}; border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-weight: 600; font-size: 0.95rem;">
                    {check_icon} {lesson['title']}
                </h4>
                <p style="color: #64748b; margin: 0; font-size: 0.85rem; line-height: 1.5;">{lesson['description']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Read", key=f"read_{track_key}_{lesson_idx}", use_container_width=True):
            # Mark as completed when clicked
            if lesson_idx not in st.session_state[progress_key]:
                st.session_state[progress_key].append(lesson_idx)
            # Open link in new tab via JavaScript
            st.markdown(f'<meta http-equiv="refresh" content="0; url={lesson["url"]}">', unsafe_allow_html=True)


def render_progress_bar(track_key: str, total_lessons: int, color: str):
    """Render progress bar for a learning track"""
    progress_key = f"completed_{track_key}"
    completed = len(st.session_state.get(progress_key, []))
    pct = int((completed / total_lessons) * 100)

    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-size: 0.8rem; color: #64748b;">Progress</span>
            <span style="font-size: 0.8rem; color: {color}; font-weight: 600;">{completed}/{total_lessons} completed</span>
        </div>
        <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {pct}%; border-radius: 4px; transition: width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# ============================================================================
# MAIN SHOW FUNCTION
# ============================================================================

def show():
    """Main Articles & Learn page"""

    # Back button at the top left
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="articles_back_to_home"):
            st.session_state.current_page ="Home"
            st.rerun()

    # Initialize session state
    if'bookmarks' not in st.session_state:
        st.session_state['bookmarks'] = []
    if'news_filter' not in st.session_state:
        st.session_state['news_filter'] ='All'
    if'topic_filter' not in st.session_state:
        st.session_state['topic_filter'] = None

    # Breadcrumb navigation
    st.markdown("""
    <div style="font-family:'Poppins', sans-serif; color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
        <span style="color: #4f8df9;">Home</span> › <span style="color: #1e293b; font-weight: 500;">Articles / Learn</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4f8df9, #4752fa); padding: 42px; border-radius: 18px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(79, 141, 249, 0.3);">
        <h1 style="font-family:'Poppins', sans-serif; font-size: 2.75rem; font-weight: 700; color: #ffffff; margin: 0 0 0.5rem 0;"> Articles & Learning Hub</h1>
        <p style="font-family:'Poppins', sans-serif; font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0;">Stay informed with categorized market news and structured learning paths</p>
    </div>
    """, unsafe_allow_html=True)

    # Tab layout
    tab1, tab2 = st.tabs([" Market News"," Learning Resources"])

    # ==========================================================================
    # TAB 1: MARKET NEWS
    # ==========================================================================
    with tab1:
        # Fetch news
        with st.spinner("Loading latest market news..."):
            articles = get_market_news()

        if articles:
            # Extract trending topics
            trending = extract_trending_topics(articles)

            # Trending Topics Section
            if trending:
                st.markdown("#### Trending Topics")

                # Create clickable topic tags
                topic_html ='<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">'
                for topic in trending:
                    is_active = st.session_state.get('topic_filter') == topic
                    bg_color ="#3b82f6" if is_active else"#f1f5f9"
                    text_color ="#ffffff" if is_active else"#475569"
                    topic_html += f'<span style="background: {bg_color}; color: {text_color}; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; cursor: pointer;">#{topic}</span>'
                topic_html +='</div>'
                st.markdown(topic_html, unsafe_allow_html=True)

                # Topic filter buttons
                topic_cols = st.columns(min(len(trending), 5))
                for i, topic in enumerate(trending[:5]):
                    with topic_cols[i]:
                        if st.button(f"#{topic}", key=f"topic_{topic}"):
                            if st.session_state['topic_filter'] == topic:
                                st.session_state['topic_filter'] = None
                            else:
                                st.session_state['topic_filter'] = topic
                            st.rerun()

                if st.session_state['topic_filter']:
                    if st.button("Clear topic filter"):
                        st.session_state['topic_filter'] = None
                        st.rerun()

            st.markdown("---")

            # Bookmarked Articles Expander
            bookmark_count = len(st.session_state['bookmarks'])
            with st.expander(f" My Bookmarked Articles ({bookmark_count})", expanded=False):
                if st.session_state['bookmarks']:
                    for i, bookmark in enumerate(st.session_state['bookmarks']):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**[{bookmark['title'][:60]}...]({bookmark['link']})** - {bookmark['source']}")
                        with col2:
                            if st.button("️", key=f"remove_bm_{i}"):
                                st.session_state['bookmarks'].pop(i)
                                st.rerun()
                else:
                    st.info("No bookmarked articles yet. Click the ️ button on any article to bookmark it.")

            # Category Sub-tabs
            category_tabs = st.tabs([" ALL","MARKET"," FED POLICY"," EARNINGS"," CRYPTO"," TECH"])

            category_map = {
                0: "All",
                1: "Market Updates",
                2: "Fed Policy",
                3: "Earnings",
                4: "Crypto",
                5: "Tech"
            }

            for tab_idx, cat_tab in enumerate(category_tabs):
                with cat_tab:
                    category = category_map[tab_idx]

                    # Filter articles
                    if category == "All":
                        filtered_articles = articles
                    else:
                        filtered_articles = [a for a in articles if a['category'] == category]

                    # Apply topic filter if active
                    if st.session_state['topic_filter']:
                        topic = st.session_state['topic_filter'].lower()
                        filtered_articles = [
                            a for a in filtered_articles
                            if topic in a['title'].lower() or topic in a.get('full_summary','').lower()
                        ]

                    if filtered_articles:
                        st.caption(f"Showing {len(filtered_articles)} articles")
                        for idx, article in enumerate(filtered_articles):
                            render_news_card(article, index=idx)
                    else:
                        st.info(f"No articles found in this category{' for topic #' + st.session_state['topic_filter'] if st.session_state['topic_filter'] else''}.")
        else:
            st.warning("Unable to load news at this time. Please check your internet connection and try again.")

    # ==========================================================================
    # TAB 2: LEARNING RESOURCES
    # ==========================================================================
    with tab2:
        st.markdown("### Choose Your Learning Path")
        st.caption("Structured curriculum from beginner to advanced. Track your progress as you complete lessons.")

        st.markdown("---")

        # Render each learning track
        for track_key, track in LEARNING_PATHS.items():
            with st.expander(track['title'], expanded=(track_key =="beginner")):
                st.markdown(f"<p style='color: #64748b; margin-bottom: 1rem;'>{track['description']}</p>", unsafe_allow_html=True)

                # Progress bar
                render_progress_bar(track_key, len(track['lessons']), track['color'])

                # Lessons
                for idx, lesson in enumerate(track['lessons']):
                    render_learning_lesson(lesson, track_key, idx, track['color'])

                # Reset progress button
                if st.button(f"Reset Progress", key=f"reset_{track_key}"):
                    st.session_state[f"completed_{track_key}"] = []
                    st.rerun()

        st.markdown("---")

        # Additional Resources
        st.markdown("### Additional Resources")

        resource_col1, resource_col2, resource_col3 = st.columns(3)

        with resource_col1:
            st.markdown("""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.25rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;"></div>
                <h4 style="margin: 0 0 0.5rem 0; color: #166534;">Investopedia</h4>
                <p style="color: #15803d; font-size: 0.85rem; margin: 0;">Comprehensive financial dictionary and tutorials</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("Visit Investopedia","https://www.investopedia.com", use_container_width=True)

        with resource_col2:
            st.markdown("""
            <div style="background: #fef3c7; border: 1px solid #fcd34d; border-radius: 12px; padding: 1.25rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;"></div>
                <h4 style="margin: 0 0 0.5rem 0; color: #92400e;">Yahoo Finance</h4>
                <p style="color: #a16207; font-size: 0.85rem; margin: 0;">Real-time market data and analysis</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("Visit Yahoo Finance","https://finance.yahoo.com", use_container_width=True)

        with resource_col3:
            st.markdown("""
            <div style="background: #ede9fe; border: 1px solid #c4b5fd; border-radius: 12px; padding: 1.25rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;"></div>
                <h4 style="margin: 0 0 0.5rem 0; color: #5b21b6;">Khan Academy</h4>
                <p style="color: #6d28d9; font-size: 0.85rem; margin: 0;">Free video courses on finance & economics</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("Visit Khan Academy","https://www.khanacademy.org/economics-finance-domain", use_container_width=True)


if __name__ =="__main__":
    show()

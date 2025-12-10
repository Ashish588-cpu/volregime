"""
VolRegime - Financial Intelligence Platform
Main app entry point with navigation and routing
"""

import streamlit as st
import sys
import os

# need to add parent dir to path so we can import from other folders
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# page config has to come before any other streamlit commands
st.set_page_config(
    page_title="VolRegime",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# hide sidebar completely and streamlit's default nav
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# apply the neon color theme (just changes colors, doesn't break anything)
from styles.neon_patch import apply_neon_patch
apply_neon_patch()

# auth stuff - login/signup functionality
from utils.auth import init_session_state, is_authenticated, render_auth_ui, render_user_menu


def apply_theme_styles():
    """global theme styles"""
    st.markdown("""
    <style>
    /* Page animation */
    .main .block-container {
        animation: page-fade-in 0.3s ease;
    }

    @keyframes page-fade-in {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Fix dropdown styling */
    .stSelectbox > div > div {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
        border: 1px solid #00e6f6 !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #00e6f6 !important;
        box-shadow: 0 0 10px rgba(0,230,246,0.4) !important;
    }

    /* Dropdown menu items */
    div[data-baseweb="select"] > div {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
    }

    div[data-baseweb="menu"] {
        background-color: #0b1220 !important;
    }

    div[data-baseweb="menu"] li {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
    }

    div[data-baseweb="menu"] li:hover {
        background-color: #1a2332 !important;
        border-left: 2px solid #00e6f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def apply_nav_styles():
    """styles for navigation bar"""
    st.markdown("""
    <style>
    /* Navigation styling */
    .stButton > button {
        font-weight: 700 !important;
    }
    div[data-testid="column"] > div > div > button[kind="primary"] {
        background-color: #8b5cf6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }
    div[data-testid="column"] > div > div > button[kind="secondary"] {
        background-color: transparent !important;
        color: #8b5cf6 !important;
        font-weight: 700 !important;
        border: 1px solid #8b5cf6 !important;
    }
    div[data-testid="column"] > div > div > button[kind="secondary"]:hover {
        background-color: rgba(139,92,246,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_top_nav(current_page: str):
    """renders the top navigation bar"""
    apply_nav_styles()

    pages = [
        "Home",
        "Market Overview",
        "Asset Dashboard",
        "Volatility Regime",
        "My Portfolio",
        "Portfolio Tools",
        "Articles / Learn",
        "Guide"
    ]

    cols = st.columns(len(pages))
    for idx, page in enumerate(pages):
        with cols[idx]:
            if st.button(page, key=f"nav_{page}", use_container_width=True,
                        type="primary" if page == current_page else "secondary"):
                st.session_state.current_page = page
                st.rerun()


def main():
    """main function that runs the whole app"""

    # setup auth stuff (checks if user is logged in)
    init_session_state()

    # apply theme styles
    apply_theme_styles()

    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Define main pages
    pages = [
        "Home",
        "Market Overview",
        "Asset Dashboard",
        "Volatility Regime",
        "My Portfolio",
        "Portfolio Tools",
        "Articles / Learn",
        "Guide",
        "About"
    ]

    # Ensure current page is valid
    if st.session_state.current_page not in pages:
        st.session_state.current_page = "Home"

    page = st.session_state.current_page

    # Auth section and navigation at top if not on Auth page
    if page != "Auth":
        auth_col1, auth_col2 = st.columns([5, 1])
        with auth_col2:
            if is_authenticated():
                user = st.session_state.get('user')
                if user:
                    st.markdown(f"**{user['email'].split('@')[0]}**")
                    if st.button("Sign Out", key="signout_btn", use_container_width=True):
                        from utils.auth import sign_out
                        sign_out()
                        st.rerun()
            else:
                if st.button("Login", key="auth_btn", use_container_width=True, type="primary"):
                    st.session_state.current_page = "Auth"
                    st.rerun()

        st.markdown("---")

        # Render top navigation
        render_top_nav(page)

        st.markdown("---")

    # Main content area - Route to pages
    if page == "Home":
        try:
            from pages.Home import show
            show()
        except Exception as e:
            st.error(f"Error loading Home: {e}")
            st.info("Home page is being updated. Please check back soon.")

    elif page == "Market Overview":
        try:
            from pages.MarketOverview import show
            show()
        except Exception as e:
            st.error(f"Error loading Market Overview: {e}")
            st.info("Market Overview page is being updated. Please check back soon.")

    elif page == "Asset Dashboard":
        try:
            from pages.AssetDashboard import show
            show()
        except Exception as e:
            st.error(f"Error loading Asset Dashboard: {e}")
            st.info("Asset Dashboard page is being updated. Please check back soon.")

    elif page == "Volatility Regime":
        try:
            from pages.VolatilityRegime import show
            show()
        except Exception as e:
            st.error(f"Error loading Volatility Regime: {e}")
            st.info("Volatility Regime page is being updated. Please check back soon.")

    elif page == "My Portfolio":
        try:
            from pages.MyPortfolio import show
            show()
        except Exception as e:
            st.error(f"Error loading My Portfolio: {e}")
            st.info("My Portfolio page is being updated. Please check back soon.")

    elif page == "Portfolio Tools":
        try:
            from pages.PortfolioTools import show
            show()
        except Exception as e:
            st.error(f"Error loading Portfolio Tools: {e}")
            st.info("Portfolio Tools page is being updated. Please check back soon.")

    elif page == "Articles / Learn":
        try:
            from pages.ArticlesLearn import show
            show()
        except Exception as e:
            st.error(f"Error loading Articles / Learn: {e}")
            st.info("Articles / Learn page is being updated. Please check back soon.")

    elif page == "Guide":
        try:
            from pages.Guide import show
            show()
        except Exception as e:
            st.error(f"Error loading Guide: {e}")
            st.info("User Guide page is being updated. Please check back soon.")

    elif page == "About":
        try:
            from pages.About import show
            show()
        except Exception as e:
            st.error(f"Error loading About: {e}")
            st.info("About page is being updated. Please check back soon.")

    elif page == "Auth":
        try:
            from pages.Auth import show
            show()
        except Exception as e:
            st.error(f"Error loading Auth: {e}")
            st.info("Authentication page is being updated. Please check back soon.")


if __name__ == "__main__":
    main()

"""
VolRegime - Minimal startup version for Streamlit Cloud
This version has ultra-fast startup to pass health checks
"""

import streamlit as st

st.set_page_config(
 page_title="VolRegime",
 layout="wide",
 initial_sidebar_state="collapsed"
)

# Fast startup message
st.title("VolRegime - Financial Intelligence Platform")
st.success("App is running! Loading full interface...")

# Lazy load everything else
def load_full_app():
 import sys
 import os

 # Add frontend_v2 to path
 frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'frontend_v2')
 sys.path.insert(0, frontend_dir)

 # Import main app
 from app import main
 main()

# Load full app after initial render
if st.button("Load Full Application") or st.session_state.get('loaded', False):
 st.session_state.loaded = True
 load_full_app()
else:
 st.info(" Click above to load the full application interface")
 st.markdown("""
 ### Quick Start
 - View real-time market data
 - Analyze volatility regimes
 - Manage your portfolio
 - Access educational resources
""")

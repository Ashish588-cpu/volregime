"""
Navigation bar component for VolRegime frontend
"""

import streamlit as st

def render_navbar():
    """Render the main navigation bar"""

    st.markdown("""
    <div style="background-color: white; padding: 1rem 2rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center;">
            <h1 style="color: #22c55e; font-size: 1.5rem; font-weight: 700; margin: 0; margin-right: 2rem;"> VolRegime</h1>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="color: #6b7280; font-size: 0.875rem;">Real-time Market Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

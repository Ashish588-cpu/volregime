"""
Loading skeleton components for VolRegime Financial Analytics Platform
Modern shimmer loading states that show content structure before data arrives
"""

import streamlit as st


# Shared shimmer animation CSS (injected once per page)
SHIMMER_CSS = """
<style>
@keyframes shimmer {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}
.skeleton-shimmer {
    animation: shimmer 1.5s ease-in-out infinite;
}
</style>
"""


def inject_shimmer_css():
    """Inject shimmer animation CSS once per page"""
    if 'shimmer_css_injected' not in st.session_state:
        st.markdown(SHIMMER_CSS, unsafe_allow_html=True)
        st.session_state.shimmer_css_injected = True


def skeleton_card(height: str = "auto", show_subtitle: bool = True) -> str:
    """
    Renders a shimmer loading skeleton for metric cards
    
    Args:
        height (str): Custom height for the skeleton card
        show_subtitle (bool): Whether to show the subtitle placeholder
    
    Returns:
        str: HTML string for the skeleton card
    """
    subtitle_html = '<div style="height: 16px; width: 30%; background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%); border-radius: 4px; background-size: 200% 100%; animation: shimmer 1.5s ease-in-out infinite;"></div>' if show_subtitle else ''
    
    skeleton_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); height: {height};">
        <div style="height: 12px; width: 40%; background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%); border-radius: 4px; margin-bottom: 1rem; background-size: 200% 100%; animation: shimmer 1.5s ease-in-out infinite;"></div>
        <div style="height: 32px; width: 60%; background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%); border-radius: 4px; margin-bottom: 0.75rem; background-size: 200% 100%; animation: shimmer 1.5s ease-in-out infinite;"></div>
        {subtitle_html}
    </div>
    """
    return skeleton_html


def skeleton_chart(height: int = 400, message: str = "Loading chart...") -> str:
    """
    Renders a shimmer loading skeleton for charts
    
    Args:
        height (int): Height of the skeleton chart in pixels
        message (str): Loading message to display
    
    Returns:
        str: HTML string for the skeleton chart
    """
    skeleton_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem; height: {height}px; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <div style="width: 48px; height: 48px; border: 3px solid #e2e8f0; border-top-color: #4f8df9; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 1rem;"></div>
        <p style="color: #94a3b8; font-family: 'Poppins', sans-serif; font-size: 0.875rem; margin: 0;">{message}</p>
    </div>
    <style>
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    return skeleton_html


def skeleton_table(rows: int = 5, cols: int = 4) -> str:
    """
    Renders a shimmer loading skeleton for data tables
    
    Args:
        rows (int): Number of skeleton rows
        cols (int): Number of skeleton columns
    
    Returns:
        str: HTML string for the skeleton table
    """
    # Header row
    header_cells = ''.join([
        '<div style="height: 14px; width: 80%; background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%); border-radius: 4px; background-size: 200% 100%; animation: shimmer 1.5s ease-in-out infinite;"></div>'
        for _ in range(cols)
    ])
    
    # Data rows
    data_rows = ''
    for i in range(rows):
        cells = ''.join([
            f'<div style="flex: 1; padding: 0.75rem;"><div style="height: 12px; width: {60 + (i * 5) % 30}%; background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%); border-radius: 4px; background-size: 200% 100%; animation: shimmer 1.5s ease-in-out infinite; animation-delay: {i * 0.1}s;"></div></div>'
            for j in range(cols)
        ])
        data_rows += f'<div style="display: flex; border-bottom: 1px solid #f1f5f9;">{cells}</div>'
    
    skeleton_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <div style="display: flex; background: #f8fafc; border-bottom: 1px solid #e2e8f0; padding: 1rem;">
            {header_cells}
        </div>
        {data_rows}
    </div>
    """
    return skeleton_html


def skeleton_metric_row(count: int = 4) -> str:
    """
    Renders a row of skeleton metric cards
    
    Args:
        count (int): Number of skeleton cards in the row
    
    Returns:
        str: HTML string for the skeleton row
    """
    cards = ''.join([
        f'<div style="flex: 1; margin: 0 0.5rem;">{skeleton_card()}</div>'
        for _ in range(count)
    ])
    
    skeleton_html = f"""
    <div style="display: flex; margin: 0 -0.5rem;">
        {cards}
    </div>
    """
    return skeleton_html


def render_skeleton_cards(cols: list, count: int = 3):
    """
    Render skeleton cards in Streamlit columns
    
    Args:
        cols (list): List of Streamlit column objects
        count (int): Number of cards to render
    """
    inject_shimmer_css()
    for i in range(min(count, len(cols))):
        with cols[i]:
            st.markdown(skeleton_card(), unsafe_allow_html=True)


def render_skeleton_chart(container=None, height: int = 400, message: str = "Loading chart..."):
    """
    Render skeleton chart in Streamlit
    
    Args:
        container: Streamlit container (optional)
        height (int): Height of skeleton chart
        message (str): Loading message
    """
    inject_shimmer_css()
    html = skeleton_chart(height, message)
    if container:
        container.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


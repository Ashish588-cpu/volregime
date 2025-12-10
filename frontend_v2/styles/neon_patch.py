"""
Neon color theme patch
Just updates colors throughout the app, doesn't change any layouts or break anything
Basically makes everything look cooler with cyan/magenta neon colors
"""

import streamlit as st

# the color palette - just using simple hex codes
NEON_COLORS = {
    'bg_dark': '#0C0F13',           # page background (super dark)
    'bg_card': '#1a1f26',            # card backgrounds
    'bg_hover': '#252b34',           # when you hover over stuff
    'cyan': '#00E9FF',               # main neon color
    'magenta': '#C400FF',            # secondary neon
    'cyan_glow': 'rgba(0,233,255,0.45)',  # glow effects
    'magenta_glow': 'rgba(196,0,255,0.35)',
    'text_primary': '#f8fafc',       # main text color
    'text_secondary': '#94a3b8',     # less important text
    'text_muted': '#64748b',         # really faded text
    'border': '#2d3e5f',             # card borders
    'success': '#00ff88',            # green for gains
    'error': '#ff3366',              # red for losses
}


def apply_neon_patch():
    """
    injects CSS to update all the colors
    doesn't mess with any layouts or functionality, just makes things look neon
    """
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== BACKGROUND COLORS ONLY ===== */
    .main {{
        background-color: {NEON_COLORS['bg_dark']};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NEON_COLORS['bg_dark']} 0%, #0f1419 100%);
    }}

    /* ===== TEXT COLORS ONLY ===== */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {NEON_COLORS['text_primary']} !important;
    }}

    /* ===== NEON ACCENTS ON EXISTING COMPONENTS ===== */

    /* Buttons - only update colors, keep existing behavior */
    .stButton > button {{
        background: linear-gradient(135deg, {NEON_COLORS['cyan']}, {NEON_COLORS['magenta']});
        border: none;
        color: {NEON_COLORS['bg_dark']};
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        box-shadow: 0 0 20px {NEON_COLORS['cyan_glow']};
        transform: translateY(-2px);
    }}

    /* Tabs - only update colors */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {NEON_COLORS['bg_card']};
        border: 1px solid {NEON_COLORS['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {NEON_COLORS['text_secondary']};
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {NEON_COLORS['bg_hover']};
        color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 10px {NEON_COLORS['cyan_glow']};
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0,233,255,0.2), rgba(196,0,255,0.2));
        color: {NEON_COLORS['cyan']};
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 20px {NEON_COLORS['cyan_glow']};
    }}

    /* Metrics - only update colors */
    [data-testid="stMetricValue"] {{
        color: {NEON_COLORS['cyan']};
        font-family: 'JetBrains Mono', monospace;
    }}

    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Navigation - only update colors */
    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: rgba(0,233,255,0.1);
        color: {NEON_COLORS['cyan']};
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 10px {NEON_COLORS['cyan_glow']};
    }}

    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
        background: linear-gradient(135deg, rgba(0,233,255,0.2), rgba(196,0,255,0.2));
        color: {NEON_COLORS['cyan']};
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 20px {NEON_COLORS['cyan_glow']};
    }}

    /* Inputs - only update colors */
    .stTextInput input, .stSelectbox select {{
        background-color: {NEON_COLORS['bg_card']};
        border: 1px solid {NEON_COLORS['border']};
        color: {NEON_COLORS['text_primary']};
    }}

    .stTextInput input:focus, .stSelectbox select:focus {{
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 10px {NEON_COLORS['cyan_glow']};
    }}

    /* Cards - only update colors on existing card classes */
    .stat-card, .asset-card, .metric-card {{
        background: {NEON_COLORS['bg_card']};
        border: 1px solid {NEON_COLORS['border']};
        transition: all 0.2s ease;
    }}

    .stat-card:hover, .asset-card:hover, .metric-card:hover {{
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 15px {NEON_COLORS['cyan_glow']};
    }}

    /* Plotly Charts - update background only */
    .js-plotly-plot .plotly {{
        background-color: {NEON_COLORS['bg_card']} !important;
    }}

    /* Dataframes - only update colors */
    .stDataFrame {{
        background-color: {NEON_COLORS['bg_card']};
    }}

    /* Expander - only update colors */
    .streamlit-expanderHeader {{
        background-color: {NEON_COLORS['bg_card']};
        border: 1px solid {NEON_COLORS['border']};
        color: {NEON_COLORS['text_primary']};
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {NEON_COLORS['cyan']};
        box-shadow: 0 0 10px {NEON_COLORS['cyan_glow']};
    }}

    /* Hover glow effect on all interactive elements */
    .stButton, .stSelectbox, .stTextInput, [data-testid="stMetric"] {{
        transition: all 0.2s ease;
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}

    ::-webkit-scrollbar-track {{
        background: {NEON_COLORS['bg_dark']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {NEON_COLORS['cyan']};
        border-radius: 5px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {NEON_COLORS['magenta']};
        box-shadow: 0 0 10px {NEON_COLORS['cyan_glow']};
    }}
    </style>
    """, unsafe_allow_html=True)


def neon_glow_animation():
    """Add subtle glow animation CSS"""
    st.markdown("""
    <style>
    @keyframes subtle-glow {
        0%, 100% { box-shadow: 0 0 10px rgba(0,233,255,0.3); }
        50% { box-shadow: 0 0 20px rgba(0,233,255,0.5); }
    }

    .glow-effect {
        animation: subtle-glow 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

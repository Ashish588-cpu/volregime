"""
Volatility Regime Detection
Upcoming ML-powered feature for regime classification
"""

import streamlit as st

def show():
    """Shows coming soon page with feature details"""

    # Hero section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0b1220, #1a1a1d);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #00e6f6;
        box-shadow: 0 0 25px rgba(0,230,246,0.3);
    ">
        <h1 style="
            font-family:'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00e6f6, #ff2b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 1rem 0;
        ">Volatility Regime Detection</h1>
        <p style="
            font-family:'Inter', sans-serif;
            color: #8b9db8;
            font-size: 1.2rem;
            margin: 0;
        ">
            ML-Powered Market State Classification • Coming Q2 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features section
    st.markdown("""
    <h2 style="
        color: #00e6f6;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    ">Upcoming Features</h2>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> XGBoost Classifier</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Supervised learning model trained on historical market data to classify current volatility regime into Low, Medium, High, or Extreme states.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> Real-time Predictions</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Live regime classification with confidence scores, updated continuously as market conditions change.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> Regime Shift Detection</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Early warning system to detect potential volatility regime transitions before they fully materialize.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> LSTM Neural Network</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Deep learning model for sequential pattern recognition, capturing complex temporal dependencies in market volatility.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> Historical Analysis</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Backtesting framework with walk-forward validation to evaluate model performance across different market cycles.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background: #0b1220;
            border: 1px solid #00e6f6;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 15px rgba(0,230,246,0.2);
        ">
            <h3 style="color: #00e6f6; margin-bottom: 1rem;"> Portfolio Optimization</h3>
            <p style="color: #e2e8f0; line-height: 1.6;">
                Regime-aware allocation strategies that automatically adjust portfolio risk based on detected market states.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technical details section
    st.markdown("""
    <h2 style="
        color: #00e6f6;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    ">Technical Approach</h2>
    """, unsafe_allow_html=True)

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0b1220, #1a1a1d);
            border: 1px solid #ff2b6b;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 15px rgba(255,43,107,0.2);
        ">
            <h3 style="color: #ff2b6b; margin-bottom: 1rem;">70+ Features</h3>
            <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.6;">
                Returns, momentum, RSI, ATR, VIX correlation, term spreads, and macro indicators
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0b1220, #1a1a1d);
            border: 1px solid #ff2b6b;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 15px rgba(255,43,107,0.2);
        ">
            <h3 style="color: #ff2b6b; margin-bottom: 1rem;">Ensemble Model</h3>
            <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.6;">
                Combining XGBoost and LSTM predictions for robust classification
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tech_col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0b1220, #1a1a1d);
            border: 1px solid #ff2b6b;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 15px rgba(255,43,107,0.2);
        ">
            <h3 style="color: #ff2b6b; margin-bottom: 1rem;">Real-time API</h3>
            <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.6;">
                FastAPI endpoint serving predictions with sub-second latency
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Timeline
    st.markdown("""
    <h2 style="
        color: #00e6f6;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    ">Development Roadmap</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0b1220, #1a1a1d);
        border: 1px solid #00e6f6;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 0 20px rgba(0,230,246,0.2);
    ">
        <div style="color: #e2e8f0; line-height: 2;">
            <p><strong style="color: #00e6f6;">Q1 2026:</strong> Feature engineering pipeline and data collection</p>
            <p><strong style="color: #00e6f6;">Q2 2026:</strong> XGBoost model training and backtesting</p>
            <p><strong style="color: #00e6f6;">Q3 2026:</strong> LSTM implementation and ensemble model</p>
            <p><strong style="color: #00e6f6;">Q4 2026:</strong> Production deployment and real-time API</p>
            <p><strong style="color: #00e6f6;">2026+:</strong> RL-based allocation agent and institutional tools</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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


if __name__ =="__main__":
    st.set_page_config(
        page_title="VolRegime - Volatility Regime",
        layout="wide"
    )
    show()

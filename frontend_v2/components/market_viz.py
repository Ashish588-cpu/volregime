"""
Market Visualization Orchestrator
=================================

Multi-panel visualization system for augmented market data:
- Top Panel: Candlestick with MA overlays, earnings markers
- Middle Panel: Volatility regime bands with cluster coloring
- Bottom Panel: Volume bars + VWAP + anomaly markers

Features:
- Dynamic crosshair synchronization
- Toggle layers (macro correlations, risk heatmap)
- Sentiment overlay as candle color gradient

Integrates with market_augmentation.py output.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import streamlit as st

# ============================================================
# CONSTANTS & STYLING
# ============================================================

REGIME_COLORS = {
'Low': {'bg':'rgba(34, 197, 94, 0.15)','border':'#22c55e','text':'#166534'},
'Medium': {'bg':'rgba(234, 179, 8, 0.15)','border':'#eab308','text':'#854d0e'},
'High': {'bg':'rgba(249, 115, 22, 0.15)','border':'#f97316','text':'#9a3412'},
'Extreme': {'bg':'rgba(239, 68, 68, 0.15)','border':'#ef4444','text':'#991b1b'},
}

CANDLE_COLORS = {
'bullish':'#22c55e',
'bearish':'#ef4444',
'bullish_line':'#16a34a',
'bearish_line':'#dc2626'
}

MA_COLORS = {
'MA20':'#3b82f6',
'MA50':'#f59e0b', 
'MA200':'#8b5cf6'
}

CHART_CONFIG = {
'displayModeBar': True,
'modeBarButtonsToRemove': ['pan2d','lasso2d','select2d'],
'displaylogo': False,
'scrollZoom': True,
'responsive': True
}

LAYOUT_DEFAULTS = {
'font': {'family':'Poppins, sans-serif','size': 12},
'paper_bgcolor':'rgba(0,0,0,0)',
'plot_bgcolor':'rgba(250, 250, 250, 1)',
'margin': {'l': 60,'r': 40,'t': 60,'b': 40},
'hovermode':'x unified',
'legend': {
'orientation':'h',
'yanchor':'bottom',
'y': 1.02,
'xanchor':'right',
'x': 1,
'bgcolor':'rgba(255,255,255,0.8)'
 }
}


# ============================================================
# PANEL 1: CANDLESTICK WITH OVERLAYS
# ============================================================

def create_candlestick_trace(df: pd.DataFrame, sentiment_overlay: bool = False) -> List[go.Candlestick]:
"""
 Create candlestick trace with optional sentiment coloring.
 
 Args:
 df: Augmented DataFrame with OHLCV
 sentiment_overlay: Color candles by sentiment
 
 Returns:
 List of candlestick traces
"""
 if sentiment_overlay and'sentiment_compound' in df.columns:
 # Color gradient based on sentiment (-1 to +1)
 sentiment = df['sentiment_compound'].fillna(0)
 increasing_colors = [
 f'rgba({int(34 + 100 * max(0, s))}, {int(197 - 50 * max(0, -s))}, 94, 0.9)'
 for s in sentiment
 ]
 decreasing_colors = [
 f'rgba({int(239 - 50 * max(0, s))}, {int(68 + 50 * max(0, s))}, 68, 0.9)'
 for s in sentiment
 ]
 else:
 increasing_colors = CANDLE_COLORS['bullish']
 decreasing_colors = CANDLE_COLORS['bearish']
 
 candlestick = go.Candlestick(
 x=df.index,
 open=df['Open'],
 high=df['High'],
 low=df['Low'],
 close=df['Close'],
 name='Price',
 increasing=dict(line=dict(color=CANDLE_COLORS['bullish_line']), fillcolor=CANDLE_COLORS['bullish']),
 decreasing=dict(line=dict(color=CANDLE_COLORS['bearish_line']), fillcolor=CANDLE_COLORS['bearish']),
 showlegend=True,
 hoverinfo='x+y'
 )
 
 return candlestick


def create_ma_traces(df: pd.DataFrame, mas: List[int] = [20, 50, 200]) -> List[go.Scatter]:
"""Create moving average overlay traces."""
 traces = []
 
 for ma in mas:
 col = f'SMA_{ma}'
 if col in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df[col],
 mode='lines',
 name=f'MA{ma}',
 line=dict(color=MA_COLORS.get(f'MA{ma}','#666'), width=1.5),
 hovertemplate=f'MA{ma}: %{{y:.2f}}<extra></extra>'
 ))
 
 return traces


def create_earnings_markers(df: pd.DataFrame) -> Optional[go.Scatter]:
"""Create earnings event markers."""
 if'earnings_window' not in df.columns:
 return None
 
 # Find earnings dates (where pre_earnings_return is not NaN)
 earnings_mask = df['pre_earnings_return'].notna()
 if not earnings_mask.any():
 return None
 
 earnings_dates = df[earnings_mask]
 
 # Color by surprise direction
 colors = ['#22c55e' if d > 0 else'#ef4444' if d < 0 else'#6b7280' 
 for d in earnings_dates.get('earnings_surprise_direction', [0] * len(earnings_dates))]
 
 return go.Scatter(
 x=earnings_dates.index,
 y=earnings_dates['High'] * 1.02,
 mode='markers',
 name='Earnings',
 marker=dict(
 symbol='diamond',
 size=12,
 color=colors,
 line=dict(color='white', width=1)
 ),
 hovertemplate='<b>Earnings</b><br>Post: %{customdata:.1%}<extra></extra>',
 customdata=earnings_dates.get('post_earnings_return', [0] * len(earnings_dates))
 )


# ============================================================
# PANEL 2: VOLATILITY REGIME BANDS
# ============================================================

def create_regime_background_shapes(df: pd.DataFrame) -> List[Dict]:
"""
 Create background shapes colored by volatility regime.

 Returns list of Plotly shape dicts for regime coloring.
"""
 shapes = []

 if'regime_name' not in df.columns:
 return shapes

 # Find regime change points
 regime_changes = df['regime_name'] != df['regime_name'].shift()
 change_indices = df.index[regime_changes].tolist()

 if not change_indices:
 return shapes

 # Add end date
 change_indices.append(df.index[-1])

 for i in range(len(change_indices) - 1):
 start_date = change_indices[i]
 end_date = change_indices[i + 1]
 regime = df.loc[start_date,'regime_name']

 if pd.isna(regime) or regime not in REGIME_COLORS:
 continue

 shapes.append({
'type':'rect',
'xref':'x2',
'yref':'y2 domain',
'x0': start_date,
'x1': end_date,
'y0': 0,
'y1': 1,
'fillcolor': REGIME_COLORS[regime]['bg'],
'line': {'width': 0},
'layer':'below'
 })

 return shapes


def create_volatility_traces(df: pd.DataFrame) -> List[go.Scatter]:
"""Create volatility line traces for middle panel."""
 traces = []

 # Realized volatility
 if'realized_vol_20d' in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df['realized_vol_20d'] * 100, # Convert to percentage
 mode='lines',
 name='Realized Vol (20d)',
 line=dict(color='#3b82f6', width=2),
 fill='tozeroy',
 fillcolor='rgba(59, 130, 246, 0.1)',
 hovertemplate='RV: %{y:.1f}%<extra></extra>'
 ))

 # EWMA volatility
 if'ewma_vol' in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df['ewma_vol'] * 100,
 mode='lines',
 name='EWMA Vol',
 line=dict(color='#f59e0b', width=1.5, dash='dash'),
 hovertemplate='EWMA: %{y:.1f}%<extra></extra>'
 ))

 # VaR threshold line
 if'VaR_95' in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df['VaR_95'].abs() * 100,
 mode='lines',
 name='VaR (95%)',
 line=dict(color='#ef4444', width=1, dash='dot'),
 hovertemplate='VaR: %{y:.2f}%<extra></extra>'
 ))

 return traces


def create_regime_annotations(df: pd.DataFrame) -> List[Dict]:
"""Create regime label annotations."""
 annotations = []

 if'regime_name' not in df.columns:
 return annotations

 # Get unique regime periods
 regime_changes = df['regime_name'] != df['regime_name'].shift()
 change_indices = df.index[regime_changes].tolist()

 for idx in change_indices:
 regime = df.loc[idx,'regime_name']
 if pd.isna(regime) or regime not in REGIME_COLORS:
 continue

 annotations.append({
'x': idx,
'y': 1,
'xref':'x2',
'yref':'y2 domain',
'text': f'<b>{regime}</b>',
'showarrow': False,
'font': {'size': 10,'color': REGIME_COLORS[regime]['text']},
'bgcolor': REGIME_COLORS[regime]['bg'],
'borderpad': 3
 })

 return annotations


# ============================================================
# PANEL 3: VOLUME WITH VWAP & ANOMALIES
# ============================================================

def create_volume_traces(df: pd.DataFrame) -> List:
"""Create volume bar traces with anomaly highlighting."""
 traces = []

 # Color bars by price direction
 colors = ['#22c55e' if c >= o else'#ef4444'
 for c, o in zip(df['Close'], df['Open'])]

 # Highlight high volume days
 if'volume_zscore' in df.columns:
 colors = [
'#3b82f6' if z > 2 else c # Blue for anomalies
 for z, c in zip(df['volume_zscore'].fillna(0), colors)
 ]

 traces.append(go.Bar(
 x=df.index,
 y=df['Volume'],
 name='Volume',
 marker=dict(color=colors, opacity=0.7),
 hovertemplate='Vol: %{y:,.0f}<extra></extra>'
 ))

 return traces


def create_vwap_trace(df: pd.DataFrame) -> Optional[go.Scatter]:
"""Create VWAP line overlay."""
 if'VWAP' not in df.columns:
 return None

 return go.Scatter(
 x=df.index,
 y=df['VWAP'],
 mode='lines',
 name='VWAP',
 line=dict(color='#8b5cf6', width=2),
 yaxis='y3b', # Secondary y-axis for price scale
 hovertemplate='VWAP: $%{y:.2f}<extra></extra>'
 )


def create_volume_anomaly_markers(df: pd.DataFrame) -> Optional[go.Scatter]:
"""Create markers for volume anomalies (z-score > 2)."""
 if'volume_zscore' not in df.columns:
 return None

 anomalies = df[df['volume_zscore'].abs() > 2]
 if anomalies.empty:
 return None

 return go.Scatter(
 x=anomalies.index,
 y=anomalies['Volume'],
 mode='markers',
 name='Vol Anomaly',
 marker=dict(
 symbol='triangle-up',
 size=10,
 color='#3b82f6',
 line=dict(color='white', width=1)
 ),
 hovertemplate='<b>Anomaly</b><br>Z-Score: %{customdata:.1f}<extra></extra>',
 customdata=anomalies['volume_zscore']
 )


# ============================================================
# TOGGLE LAYERS: MACRO CORRELATIONS & RISK HEATMAP
# ============================================================

def create_macro_correlation_trace(df: pd.DataFrame) -> List[go.Scatter]:
"""Create macro correlation overlay traces."""
 traces = []

 # Beta to SPY
 if'beta_SPY' in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df['beta_SPY'],
 mode='lines',
 name='Beta (SPY)',
 line=dict(color='#06b6d4', width=1.5),
 visible='legendonly', # Toggle via legend
 yaxis='y4',
 hovertemplate='Beta: %{y:.2f}<extra></extra>'
 ))

 # TLT correlation (duration proxy)
 if'TLT_correlation' in df.columns:
 traces.append(go.Scatter(
 x=df.index,
 y=df['TLT_correlation'],
 mode='lines',
 name='TLT Corr',
 line=dict(color='#a855f7', width=1.5, dash='dash'),
 visible='legendonly',
 yaxis='y4',
 hovertemplate='TLT Corr: %{y:.2f}<extra></extra>'
 ))

 return traces


def create_risk_heatmap_data(df: pd.DataFrame) -> Dict:
"""
 Create risk factor heatmap data for display.
 Returns data structure for heatmap visualization.
"""
 risk_factors = ['realized_vol_20d','VaR_95','beta_SPY','RSI_14','momentum_score']
 available_factors = [f for f in risk_factors if f in df.columns]

 if not available_factors:
 return {}

 # Get latest values
 latest = df.iloc[-1]

 # Normalize to 0-1 scale
 heatmap_data = {}
 for factor in available_factors:
 series = df[factor].dropna()
 if len(series) > 0:
 min_val, max_val = series.min(), series.max()
 if max_val > min_val:
 normalized = (latest[factor] - min_val) / (max_val - min_val)
 else:
 normalized = 0.5
 heatmap_data[factor] = {
'value': latest[factor],
'normalized': normalized,
'percentile': (series < latest[factor]).mean()
 }

 return heatmap_data


# ============================================================
# MAIN FIGURE BUILDER
# ============================================================

def create_multi_panel_chart(
 df: pd.DataFrame,
 symbol: str ='Asset',
 show_ma: List[int] = [20, 50, 200],
 show_earnings: bool = True,
 show_regime_bands: bool = True,
 sentiment_overlay: bool = False,
 show_macro: bool = False,
 height: int = 900
) -> go.Figure:
"""
 Create multi-panel market visualization.

 Args:
 df: Augmented market DataFrame
 symbol: Asset symbol for title
 show_ma: Moving averages to display
 show_earnings: Show earnings markers
 show_regime_bands: Color background by regime
 sentiment_overlay: Color candles by sentiment
 show_macro: Show macro correlation traces
 height: Total chart height

 Returns:
 Plotly Figure object
"""
 # Create subplots: 3 rows
 fig = make_subplots(
 rows=3, cols=1,
 shared_xaxes=True,
 vertical_spacing=0.08,
 row_heights=[0.50, 0.25, 0.25],
 subplot_titles=(
 f'{symbol} Price & Technicals',
'Volatility Regime',
'Volume Analysis'
 )
 )

 # ============================
 # ROW 1: CANDLESTICK PANEL
 # ============================

 # Candlestick
 candlestick = create_candlestick_trace(df, sentiment_overlay)
 fig.add_trace(candlestick, row=1, col=1)

 # Moving averages
 for ma_trace in create_ma_traces(df, show_ma):
 fig.add_trace(ma_trace, row=1, col=1)

 # Earnings markers
 if show_earnings:
 earnings_trace = create_earnings_markers(df)
 if earnings_trace:
 fig.add_trace(earnings_trace, row=1, col=1)

 # ============================
 # ROW 2: VOLATILITY PANEL
 # ============================

 # Volatility traces
 for vol_trace in create_volatility_traces(df):
 fig.add_trace(vol_trace, row=2, col=1)

 # Regime background shapes
 if show_regime_bands:
 shapes = create_regime_background_shapes(df)
 for shape in shapes:
 # Adjust references for row 2
 shape['xref'] ='x2'
 shape['yref'] ='y2 domain'
 fig.add_shape(**shape)

 # ============================
 # ROW 3: VOLUME PANEL
 # ============================

 # Volume bars
 for vol_trace in create_volume_traces(df):
 fig.add_trace(vol_trace, row=3, col=1)

 # Volume anomaly markers
 anomaly_trace = create_volume_anomaly_markers(df)
 if anomaly_trace:
 fig.add_trace(anomaly_trace, row=3, col=1)

 # ============================
 # MACRO CORRELATION OVERLAY
 # ============================

 if show_macro:
 for macro_trace in create_macro_correlation_trace(df):
 fig.add_trace(macro_trace, row=1, col=1)

 # ============================
 # LAYOUT CONFIGURATION
 # ============================

 fig.update_layout(
 height=height,
 title=dict(
 text=f'<b>{symbol}</b> Multi-Factor Analysis',
 font=dict(size=18, family='Poppins, sans-serif'),
 x=0.5,
 xanchor='center'
 ),
 **LAYOUT_DEFAULTS,
 xaxis_rangeslider_visible=False,
 showlegend=True
 )

 # Configure axes
 fig.update_xaxes(
 showgrid=True,
 gridwidth=1,
 gridcolor='rgba(0,0,0,0.05)',
 showspikes=True,
 spikemode='across',
 spikethickness=1,
 spikecolor='#666',
 spikedash='dot'
 )

 fig.update_yaxes(
 showgrid=True,
 gridwidth=1,
 gridcolor='rgba(0,0,0,0.05)',
 showspikes=True,
 spikethickness=1
 )

 # Row-specific y-axis labels
 fig.update_yaxes(title_text='Price ($)', row=1, col=1)
 fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)
 fig.update_yaxes(title_text='Volume', row=3, col=1)

 return fig


# ============================================================
# RISK FACTOR HEATMAP COMPONENT
# ============================================================

def create_risk_heatmap_figure(df: pd.DataFrame) -> go.Figure:
"""
 Create risk factor heatmap visualization.

 Shows current risk metrics as colored tiles.
"""
 risk_data = create_risk_heatmap_data(df)

 if not risk_data:
 # Return empty figure with message
 fig = go.Figure()
 fig.add_annotation(
 text="No risk data available",
 xref="paper", yref="paper",
 x=0.5, y=0.5, showarrow=False
 )
 return fig

 # Prepare data for heatmap
 factors = list(risk_data.keys())
 values = [risk_data[f]['normalized'] for f in factors]
 percentiles = [risk_data[f]['percentile'] * 100 for f in factors]
 raw_values = [risk_data[f]['value'] for f in factors]

 # Clean factor names for display
 display_names = [
 f.replace('_','').replace('realized vol 20d','Vol 20D')
 .replace('VaR 95','VaR (95%)')
 .replace('beta SPY','Beta')
 .replace('RSI 14','RSI')
 .replace('momentum score','Momentum')
 for f in factors
 ]

 fig = go.Figure(data=go.Heatmap(
 z=[values],
 x=display_names,
 y=['Current'],
 colorscale='RdYlGn_r', # Red = high risk, Green = low
 showscale=True,
 colorbar=dict(
 title='Risk Level',
 tickvals=[0, 0.5, 1],
 ticktext=['Low','Med','High']
 ),
 hovertemplate='<b>%{x}</b><br>Value: %{customdata[0]:.2f}<br>Percentile: %{customdata[1]:.0f}%<extra></extra>',
 customdata=[[rv, pct] for rv, pct in zip(raw_values, percentiles)]
 ))

 fig.update_layout(
 title='Risk Factor Heatmap',
 height=150,
 margin=dict(l=20, r=20, t=40, b=20),
 font=dict(family='Poppins, sans-serif'),
 paper_bgcolor='rgba(0,0,0,0)',
 plot_bgcolor='rgba(0,0,0,0)'
 )

 return fig


# ============================================================
# STREAMLIT INTEGRATION
# ============================================================

def render_market_dashboard(
 df: pd.DataFrame,
 symbol: str ='Asset',
 key_prefix: str ='market_viz'
) -> None:
"""
 Render complete market visualization dashboard in Streamlit.

 Args:
 df: Augmented market DataFrame
 symbol: Asset symbol
 key_prefix: Unique key prefix for Streamlit components
"""
 # Controls row
 col1, col2, col3, col4 = st.columns(4)

 with col1:
 show_ma = st.multiselect(
'Moving Averages',
 options=[20, 50, 200],
 default=[50, 200],
 key=f'{key_prefix}_ma'
 )

 with col2:
 show_earnings = st.checkbox('Show Earnings', value=True, key=f'{key_prefix}_earnings')
 show_regime = st.checkbox('Regime Bands', value=True, key=f'{key_prefix}_regime')

 with col3:
 sentiment_overlay = st.checkbox('Sentiment Overlay', value=False, key=f'{key_prefix}_sentiment')
 show_macro = st.checkbox('Macro Correlations', value=False, key=f'{key_prefix}_macro')

 with col4:
 chart_height = st.slider('Chart Height', 600, 1200, 900, 50, key=f'{key_prefix}_height')

 # Create and display main chart
 fig = create_multi_panel_chart(
 df=df,
 symbol=symbol,
 show_ma=show_ma,
 show_earnings=show_earnings,
 show_regime_bands=show_regime,
 sentiment_overlay=sentiment_overlay,
 show_macro=show_macro,
 height=chart_height
 )

 st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

 # Risk heatmap section
 with st.expander(" Risk Factor Heatmap", expanded=False):
 heatmap_fig = create_risk_heatmap_figure(df)
 st.plotly_chart(heatmap_fig, use_container_width=True)

 # Current metrics summary
 if not df.empty:
 latest = df.iloc[-1]

 st.markdown("### Current Metrics")

 metrics_cols = st.columns(6)

 with metrics_cols[0]:
 st.metric(
"Price",
 f"${latest['Close']:.2f}",
 f"{latest.get('return_1d', 0) * 100:.2f}%"
 )

 with metrics_cols[1]:
 vol = latest.get('realized_vol_20d', 0) * 100
 st.metric("Volatility", f"{vol:.1f}%")

 with metrics_cols[2]:
 rsi = latest.get('RSI_14', 50)
 st.metric("RSI", f"{rsi:.1f}")

 with metrics_cols[3]:
 regime = latest.get('regime_name','N/A')
 st.metric("Regime", regime)

 with metrics_cols[4]:
 beta = latest.get('beta_SPY', 1.0)
 st.metric("Beta (SPY)", f"{beta:.2f}" if not pd.isna(beta) else"N/A")

 with metrics_cols[5]:
 var = latest.get('VaR_95', 0) * 100
 st.metric("VaR (95%)", f"{var:.2f}%")


def export_chart_config(fig: go.Figure) -> str:
"""
 Export chart configuration as JSON.

 Args:
 fig: Plotly Figure object

 Returns:
 JSON string of chart configuration
"""
 return fig.to_json()


def get_chart_html(fig: go.Figure, include_plotlyjs: str ='cdn') -> str:
"""
 Get standalone HTML for chart embedding.

 Args:
 fig: Plotly Figure object
 include_plotlyjs: How to include Plotly.js ('cdn','inline', False)

 Returns:
 HTML string
"""
 return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)


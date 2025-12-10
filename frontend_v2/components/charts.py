"""
Chart components for VolRegime frontend
Interactive charts using Plotly and Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

def price_chart(data: pd.DataFrame, title: str = "Price Chart", height: int = 400):
    """
    Create an interactive price chart
    
    Args:
        data: DataFrame with Date and Close columns
        title: Chart title
        height: Chart height in pixels
    """
    
    if data.empty:
        st.warning("No data available for chart")
        return
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#22c55e', width=2),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        yaxis=dict(
            title='Price ($)',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def candlestick_chart(data: pd.DataFrame, title: str = "Candlestick Chart", height: int = 400):
    """
    Create a candlestick chart
    
    Args:
        data: DataFrame with OHLC data
        title: Chart title
        height: Chart height in pixels
    """
    
    if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        st.warning("OHLC data not available for candlestick chart")
        return
    
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444',
        name='Price'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#f3f4f6',
            showgrid=True,
            rangeslider_visible=False
        ),
        yaxis=dict(
            title='Price ($)',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def volume_chart(data: pd.DataFrame, title: str = "Volume Chart", height: int = 200):
    """
    Create a volume chart
    
    Args:
        data: DataFrame with Volume column
        title: Chart title
        height: Chart height in pixels
    """
    
    if data.empty or 'Volume' not in data.columns:
        st.warning("Volume data not available")
        return
    
    # Color bars based on price change
    colors = []
    if 'Close' in data.columns:
        for i in range(len(data)):
            if i == 0:
                colors.append('#6b7280')  # neutral for first bar
            else:
                if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                    colors.append('#22c55e')  # green for up
                else:
                    colors.append('#ef4444')  # red for down
    else:
        colors = ['#6b7280'] * len(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        yaxis=dict(
            title='Volume',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def volatility_chart(data: pd.DataFrame, title: str = "Volatility Analysis", height: int = 300):
    """
    Create a volatility chart (placeholder for ML model integration)
    
    Args:
        data: DataFrame with price data
        title: Chart title
        height: Chart height in pixels
    """
    
    if data.empty:
        st.warning("No data available for volatility analysis")
        return
    
    # Calculate simple rolling volatility as placeholder
    returns = data['Close'].pct_change().dropna()
    volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized volatility
    
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(go.Scatter(
        x=volatility.index,
        y=volatility,
        mode='lines',
        name='Volatility',
        line=dict(color='#f59e0b', width=2),
        fill='tonexty',
        fillcolor='rgba(245, 158, 11, 0.1)',
        hovertemplate='<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>'
    ))
    
    # Add regime zones (placeholder)
    fig.add_hline(y=20, line_dash="dash", line_color="#ef4444", 
                  annotation_text="High Volatility", annotation_position="right")
    fig.add_hline(y=10, line_dash="dash", line_color="#22c55e", 
                  annotation_text="Low Volatility", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        yaxis=dict(
            title='Volatility (%)',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def risk_metrics_chart(metrics: Dict, title: str = "Risk Metrics", height: int = 300):
    """
    Create a risk metrics visualization (placeholder for ML model integration)
    
    Args:
        metrics: Dictionary of risk metrics
        title: Chart title
        height: Chart height in pixels
    """
    
    if not metrics:
        # Create sample risk metrics for demonstration
        metrics = {
            'VaR (95%)': -2.5,
            'Expected Shortfall': -3.8,
            'Sharpe Ratio': 1.24,
            'Max Drawdown': -15.2,
            'Beta': 1.18,
            'Alpha': 2.3
        }
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Color bars based on value (green for positive, red for negative)
    colors = ['#22c55e' if val >= 0 else '#ef4444' for val in metric_values]
    
    fig.add_trace(go.Bar(
        y=metric_names,
        x=metric_values,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Value: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Value',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        yaxis=dict(
            title='Metrics',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def market_comparison_chart(data: Dict, title: str = "Market Comparison", height: int = 400):
    """
    Create a market comparison chart
    
    Args:
        data: Dictionary with symbol: price_data pairs
        title: Chart title
        height: Chart height in pixels
    """
    
    if not data:
        st.warning("No data available for market comparison")
        return
    
    fig = go.Figure()
    
    colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    
    for i, (symbol, price_data) in enumerate(data.items()):
        if isinstance(price_data, pd.DataFrame) and not price_data.empty:
            # Normalize to percentage change from first value
            normalized = (price_data['Close'] / price_data['Close'].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized,
                mode='lines',
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{symbol}</b><br>%{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        yaxis=dict(
            title='Change (%)',
            gridcolor='#f3f4f6',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def mini_sparkline(values: List[float], color: str = "#22c55e", height: int = 50):
    """
    Create a mini sparkline chart
    
    Args:
        values: List of values to plot
        color: Line color
        height: Chart height in pixels
    """
    
    if not values:
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(values))),
        y=values,
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def advanced_indicators_chart(data: pd.DataFrame, indicators: Dict, title: str = "Advanced Indicators"):
    """Create chart with multiple technical indicators"""
    if data.empty:
        return None

    # Create subplots
    rows = 1 + len([k for k in indicators.keys() if k in ['RSI', 'MACD', 'Volume']])
    subplot_titles = [title]

    if 'RSI' in indicators and indicators['RSI']:
        subplot_titles.append('RSI')
    if 'MACD' in indicators and indicators['MACD']:
        subplot_titles.append('MACD')
    if 'Volume' in indicators and indicators['Volume']:
        subplot_titles.append('Volume')

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.4/(rows-1)]*(rows-1) if rows > 1 else [1.0],
        vertical_spacing=0.05
    )

    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add moving averages
    if 'SMA_20' in indicators and indicators['SMA_20'] and 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )

    if 'SMA_50' in indicators and indicators['SMA_50'] and 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )

    # Bollinger Bands
    if 'BB' in indicators and indicators['BB'] and all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )

    current_row = 2

    # RSI
    if 'RSI' in indicators and indicators['RSI'] and 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1

    # MACD
    if 'MACD' in indicators and indicators['MACD'] and all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=current_row, col=1
        )
        if 'MACD_Histogram' in data.columns:
            fig.add_trace(
                go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', marker_color='gray'),
                row=current_row, col=1
            )
        current_row += 1

    # Volume
    if 'Volume' in indicators and indicators['Volume'] and 'Volume' in data.columns:
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
            row=current_row, col=1
        )

    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig

def sector_heatmap(sector_data: Dict, title: str = "Sector Performance"):
    """Create sector performance heatmap"""
    if not sector_data:
        return None

    sectors = list(sector_data.keys())
    changes = [sector_data[sector]['change_percent'] for sector in sectors]

    # Create color scale
    colors = []
    for change in changes:
        if change > 2:
            colors.append('#10b981')  # Strong green
        elif change > 0:
            colors.append('#86efac')  # Light green
        elif change > -2:
            colors.append('#fca5a5')  # Light red
        else:
            colors.append('#ef4444')  # Strong red

    fig = go.Figure(data=go.Bar(
        x=sectors,
        y=changes,
        marker_color=colors,
        text=[f"{change:.1f}%" for change in changes],
        textposition='auto'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Sectors",
        yaxis_title="Change %",
        height=400,
        showlegend=False
    )

    return fig

def volatility_cone_chart(data: pd.DataFrame, title: str = "Volatility Cone"):
    """Create volatility cone visualization"""
    if data.empty or 'Close' not in data.columns:
        return None

    returns = data['Close'].pct_change().dropna()

    # Calculate volatility for different periods
    periods = [5, 10, 20, 30, 60, 90, 120, 252]
    volatilities = []

    for period in periods:
        if len(returns) >= period:
            vol_series = returns.rolling(period).std() * np.sqrt(252)
            volatilities.append({
                'period': period,
                'min': vol_series.min(),
                'max': vol_series.max(),
                'median': vol_series.median(),
                'current': vol_series.iloc[-1]
            })

    if not volatilities:
        return None

    periods_list = [v['period'] for v in volatilities]
    min_vols = [v['min'] for v in volatilities]
    max_vols = [v['max'] for v in volatilities]
    median_vols = [v['median'] for v in volatilities]
    current_vols = [v['current'] for v in volatilities]

    fig = go.Figure()

    # Add volatility cone
    fig.add_trace(go.Scatter(
        x=periods_list, y=max_vols,
        mode='lines',
        name='Max',
        line=dict(color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=periods_list, y=min_vols,
        mode='lines',
        name='Min',
        line=dict(color='green', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=periods_list, y=median_vols,
        mode='lines+markers',
        name='Median',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=periods_list, y=current_vols,
        mode='lines+markers',
        name='Current',
        line=dict(color='orange', width=3)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Period (Days)",
        yaxis_title="Annualized Volatility",
        height=400,
        showlegend=True
    )

    return fig

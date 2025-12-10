"""
Professional Card Components for VolRegime Financial Analytics Platform

This module provides reusable UI card components with consistent styling
and professional fintech-grade design patterns.

Components:
 - metric_card: Financial metrics with change indicators
 - info_card: General information display
 - status_card: Status indicators with color coding
 - compact_metric_card: Dark theme compact metric cards for dashboards
"""

import streamlit as st


def metric_card(title, value, change=None, change_type="neutral"):
"""
 Renders a professional financial metric card with optional change indicator.

 Features:
 - Automatic number formatting (K, M, B suffixes)
 - Color-coded change indicators
 - Hover effects for interactivity
 - Responsive design

 Args:
 title (str): The metric title/label to display
 value (str|float|int): The primary metric value
 change (str, optional): Change indicator text (e.g.,"+5.2%")
 change_type (str): Color theme -"positive","negative", or"neutral"

 Example:
 metric_card("S&P 500", 4150.25,"+1.2%","positive")
"""

 # Smart number formatting with financial conventions
 if isinstance(value, (int, float)):
 if abs(value) >= 1e9:
 formatted_value = f"${value/1e9:.2f}B"
 elif abs(value) >= 1e6:
 formatted_value = f"${value/1e6:.2f}M"
 elif abs(value) >= 1e3:
 formatted_value = f"${value/1e3:.2f}K"
 else:
 formatted_value = f"${value:,.2f}"
 else:
 formatted_value = str(value)

 # Professional color palette for financial data
 color_scheme = {
"positive":"#059669",
"negative":"#dc2626",
"neutral":"#64748b"
 }

 change_color = color_scheme.get(change_type, color_scheme["neutral"])

 # Conditional change indicator rendering
 change_html =""
 if change:
 change_html = f'<div style="font-size: 0.875rem; color: {change_color}; font-weight: 600; margin-top: 0.25rem;">{change}</div>'

 # Render the complete metric card with hover effects
 st.markdown(f"""
 <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; transition: all 0.2s ease; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);" onmouseover="this.style.boxShadow='0 4px 6px -1px rgba(0, 0, 0, 0.1)'" onmouseout="this.style.boxShadow='0 1px 3px 0 rgba(0, 0, 0, 0.1)'">
 <div style="font-size: 0.875rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{title}</div>
 <div style="font-size: 1.875rem; font-weight: 700; color: #0f172a; line-height: 1.2;">{formatted_value}</div>
 {change_html}
 </div>
""", unsafe_allow_html=True)

def info_card(title, content, icon=None):
"""
 Renders an informational card for displaying general content.

 Features:
 - Clean, readable typography
 - Optional icon support
 - Subtle background styling
 - Consistent spacing

 Args:
 title (str): The card header/title text
 content (str): The main content body text
 icon (str, optional): Unicode emoji or icon character

 Example:
 info_card("Market Status","Markets are currently open","")
"""

 # Conditional icon rendering with proper spacing
 icon_html = f"<span style='margin-right: 0.5rem;'>{icon}</span>" if icon else""

 # Render informational card with subtle styling
 st.markdown(f"""
 <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem;">
 <div style="font-size: 1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.75rem;">{icon_html}{title}</div>
 <div style="font-size: 0.875rem; color: #475569; line-height: 1.5;">{content}</div>
 </div>
""", unsafe_allow_html=True)

def status_card(title, status, description=None):
"""
 Renders a status card with intelligent color-coded status indicators.

 Features:
 - Automatic color mapping for common financial statuses
 - Optional description text
 - Professional status typography
 - Consistent card styling

 Args:
 title (str): The status category/label
 status (str): The current status value
 description (str, optional): Additional context or details

 Example:
 status_card("Market Status","Open","Trading hours: 9:30 AM - 4:00 PM EST")
"""

 # Comprehensive status color mapping for financial contexts
 status_color_map = {
"active":"#059669","open":"#059669","closed":"#dc2626",
"low":"#059669","moderate":"#d97706","high":"#dc2626",
"calm":"#059669","normal":"#059669","elevated":"#d97706","crisis":"#dc2626"
 }

 # Get appropriate color or default to neutral gray
 status_color = status_color_map.get(status.lower(),"#64748b")

 # Conditional description rendering
 description_html =""
 if description:
 description_html = f'<div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">{description}</div>'

 # Render the complete status card
 st.markdown(f"""
 <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem;">
 <div style="font-size: 0.875rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{title}</div>
 <div style="font-size: 1.125rem; font-weight: 600; color: {status_color};">{status}</div>
 {description_html}
 </div>
""", unsafe_allow_html=True)

def compact_metric_card(label, value, change=None, change_type='neutral'):
"""
 Renders a compact dark theme metric card optimized for dashboard displays.

 Features:
 - Dark terminal-style design (#1e293b background)
 - Minimal padding for space efficiency
 - Color-coded change indicators
 - Professional financial typography

 Args:
 label (str): The metric label/name
 value (str|float|int): The metric value to display
 change (str, optional): Change indicator (e.g.,"+2.5%","-150")
 change_type (str):"positive","negative", or"neutral"

 Example:
 compact_metric_card("VIX","18.42","+1.2%","positive")
"""

 # Format numeric values for financial display
 if isinstance(value, (int, float)):
 if abs(value) >= 1e9:
 formatted_value = f"{value/1e9:.2f}B"
 elif abs(value) >= 1e6:
 formatted_value = f"{value/1e6:.2f}M"
 elif abs(value) >= 1e3:
 formatted_value = f"{value/1e3:.1f}K"
 else:
 formatted_value = f"{value:,.2f}"
 else:
 formatted_value = str(value)

 # Dark theme color palette
 change_colors = {
"positive":"#10b981",
"negative":"#ef4444",
"neutral":"#94a3b8"
 }

 change_color = change_colors.get(change_type, change_colors["neutral"])

 # Conditional change indicator
 change_html =""
 if change:
 change_html = f'<span style="font-size: 0.75rem; color: {change_color}; font-weight: 600; margin-left: 0.5rem;">{change}</span>'

 # Render compact dark theme card
 st.markdown(f"""
 <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem 1rem;">
 <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">{label}</div>
 <div style="display: flex; align-items: baseline;">
 <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">{formatted_value}</span>
 {change_html}
 </div>
 </div>
""", unsafe_allow_html=True)

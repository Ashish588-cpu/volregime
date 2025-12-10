"""
Portfolio management utilities
Handles portfolio holdings, P/L calculations, and price alerts
"""

from typing import Dict, List, Optional
from datetime import datetime
import yfinance as yf


class PortfolioManager:
    """Manages user portfolio holdings and calculations"""

    def __init__(self, user_email: str):
        self.user_email = user_email

    def calculate_position_pnl(self, ticker: str, quantity: float, purchase_price: float) -> Dict:
        """Calculate P&L for a single position"""
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1d")

            if data.empty:
                return None

            current_price = data['Close'].iloc[-1]
            total_cost = quantity * purchase_price
            current_value = quantity * current_price
            pnl = current_value - total_cost
            pnl_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0

            # Get stock info for sector
            info = ticker_obj.info
            sector = info.get('sector', 'Unknown')
            name = info.get('shortName', ticker)

            return {
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'total_cost': total_cost,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
        except Exception:
            return None

    def calculate_portfolio_metrics(self, holdings: List[Dict]) -> Dict:
        """Calculate overall portfolio metrics"""
        total_cost = 0
        total_value = 0
        sector_allocation = {}
        ticker_allocation = {}

        for holding in holdings:
            position = self.calculate_position_pnl(
                holding['ticker'],
                holding['quantity'],
                holding['purchase_price']
            )

            if position:
                total_cost += position['total_cost']
                total_value += position['current_value']

                # Track sector allocation
                sector = position['sector']
                if sector in sector_allocation:
                    sector_allocation[sector] += position['current_value']
                else:
                    sector_allocation[sector] = position['current_value']

                # Track ticker allocation
                ticker_allocation[position['ticker']] = position['current_value']

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

        return {
            'total_cost': total_cost,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'sector_allocation': sector_allocation,
            'ticker_allocation': ticker_allocation,
            'num_positions': len(holdings)
        }


class PriceAlertManager:
    """Manages price alerts for stocks"""

    def __init__(self, user_email: str):
        self.user_email = user_email

    def check_alert(self, ticker: str, target_price: float, alert_type: str) -> Dict:
        """Check if an alert condition is met"""
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1d")

            if data.empty:
                return None

            current_price = data['Close'].iloc[-1]

            triggered = False
            if alert_type == "above" and current_price >= target_price:
                triggered = True
            elif alert_type == "below" and current_price <= target_price:
                triggered = True

            return {
                'ticker': ticker,
                'current_price': current_price,
                'target_price': target_price,
                'alert_type': alert_type,
                'triggered': triggered,
                'checked_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception:
            return None

    def get_stock_info(self, ticker: str) -> Dict:
        """Get basic stock information for alert display"""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            data = ticker_obj.history(period="1d")

            if data.empty:
                return None

            return {
                'ticker': ticker,
                'name': info.get('shortName', ticker),
                'current_price': data['Close'].iloc[-1]
            }
        except Exception:
            return None

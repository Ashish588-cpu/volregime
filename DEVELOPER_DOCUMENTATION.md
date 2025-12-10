# VolRegime Developer Documentation
Ashish Dahal

This is a financial intelligence platform for market analysis, portfolio management, and volatility regime detection.

---

## 1. Architecture Overview

### System Components

**Streamlit Frontend**
The frontend is a multipage app. It uses custom navigation. Plotly handles the charts. Session state stores user data. The UI uses cyan and magenta colors.

**Python Backend**
The backend pulls data from Yahoo Finance using yfinance. It calculates technical indicators like RSI and Bollinger Bands. It runs portfolio analytics. It classifies volatility regimes.

**Data Layer**
Market data comes from yfinance in realtime. Session data persists in browser localStorage. Supabase handles authentication. Users can optionally store portfolios in Supabase.

**Alert Engine**
The system monitors prices in realtime. It sends notifications when thresholds are hit. Alert conditions are customizable.

---

## 2. Data Flow Diagram

![Data Flow Architecture](docs/images/data_flow_diagram.png)

**Flow Overview:**
1. **User** interacts with Streamlit UI (Home, Dashboard, Portfolio Tools)
2. **Backend Functions** process requests (get_stock_data, calculate_indicators, portfolio_metrics)
3. **Data API (yfinance)** fetches real-time prices, historical OHLCV, and company info
4. **UI Render** displays results via Plotly charts, metric cards, and data tables

---

## 3. Tech Stack

**Core Framework**
Python 3.12 and Streamlit.

**Data Sources**
yfinance for market data. feedparser for news.

**Visualization**
Plotly for charts. Custom HTML and CSS components.

**Storage**
Browser localStorage for session data. Supabase for auth and database.

**Libraries**
pandas for data. numpy for math. requests for API calls.

---

## 4. Core Modules

### 4.1 Ticker Dashboard

This module shows live market data. You input a ticker symbol. It outputs the current price, price change, historical chart, and volume.

```python
@st.cache_data(ttl=10)
def get_asset_data(ticker: str, period: str = "5d"):
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period=period)
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    change_pct = ((current_price - prev_close) / prev_close) * 100
    return {'price': current_price, 'change_pct': change_pct}
```

The page refreshes every 10 seconds. Cards are cyan for gains and magenta for losses.

---

### 4.2 Portfolio Management

This tracks your holdings. You enter ticker, shares, and cost basis. It calculates total value, profit and loss, position weights, and shows an allocation chart.

```python
def calculate_portfolio_metrics(portfolio_df):
    portfolio_df['Total Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
    total_value = portfolio_df['Total Value'].sum()
    portfolio_df['P&L'] = portfolio_df['Total Value'] - portfolio_df['Total Cost']
    return {'total_value': total_value, 'total_pnl': portfolio_df['P&L'].sum()}
```

You can add or remove positions. Values update in realtime. You can download your portfolio as CSV.

---

### 4.3 Risk Metrics

This calculates risk measures for your portfolio. Input is your holdings and price history. Output is Sharpe Ratio, Sortino Ratio, Maximum Drawdown, volatility, and correlation matrix.

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices):
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

Metrics show side by side. Tooltips explain each one. Colors indicate risk levels.

---

### 4.4 Volatility Regime Detector

This classifies the current market state. Input is VIX level, rolling standard deviation, and historical volatility. Output is regime classification (Low, Medium, High, Extreme).

```python
def classify_regime(vix_level):
    if vix_level < 15: return "Low"
    elif vix_level < 20: return "Medium"
    elif vix_level < 30: return "High"
    else: return "Extreme"
```

A visual indicator shows the current regime. Charts show historical regimes. You can compare performance across different regimes.

---

### 4.5 Technical Indicators

This calculates common indicators. Input is price data (OHLCV) and parameters. Output includes RSI, MACD, Bollinger Bands, and EMAs.

```python
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

Indicators overlay on price charts. You can toggle them on or off.

---

### 4.6 Macro Panel

This tracks macro indicators. It monitors Treasury tickers, Dollar Index, VIX, and Crude Oil. It shows current values, daily changes, and correlation with your portfolio.

```python
def get_macro_snapshot():
    macro_tickers = {'TLT': '20Y Treasury', '^VIX': 'VIX', 'CL=F': 'Oil'}
    snapshot = {}
    for ticker, name in macro_tickers.items():
        snapshot[name] = get_asset_data(ticker)
    return snapshot
```

Data displays in compact cards. Colors show movements.

---

## 5. Deployment

### Streamlit Cloud

**requirements.txt:**
```
streamlit
pandas
yfinance
plotly
supabase
```

**Environment Variables:**
Put your Supabase URL and key in secrets.toml.

**Refresh Rates:**
Home page: 10 seconds. Asset Dashboard: 30 seconds. Portfolio: manual.

**Caching:**
Price data: 10 seconds. Historical data: 5 minutes. News: 5 minutes.

---

## 6. Performance

### Rate Limits

Yahoo Finance allows 2000 requests per hour. We use caching to avoid hitting limits.

```python
@st.cache_data(ttl=10)
def get_stock_data(ticker, period):
    return yf.Ticker(ticker).history(period=period)
```

### Storage

User portfolios save to session state. They persist during your session. They are lost on refresh unless saved to localStorage or Supabase.

---

## 7. Security

### API Keys

Supabase credentials live in .streamlit/secrets.toml. Never commit this file. Use environment variables in production.

### User Data

Each session is isolated. No cross user data leakage. Sessions expire when you close the browser.

### Privacy

Only email is stored (optional). No payment info. No tracking. No third party analytics.

---

## Phase V2: Machine Learning

### Supervised Learning

**Goal:** Predict volatility regime shifts before they happen.

**Features:**
Returns (1D, 5D, 20D), momentum, RSI, ATR, VIX level, term spread, SPY VIX correlation.

**Models:**
XGBoost Classifier or LSTM Neural Network.

**Training:**
Split 80/20. Use walk forward validation. Test on 12 months, predict next 3 months, roll forward.

```python
def predict_regime(features):
    model = load_model('xgboost_v1.pkl')
    prediction = model.predict_proba([features])
    return {
        'regime': classes[prediction.argmax()],
        'confidence': prediction.max()
    }
```

---

## Roadmap

### Q1 2026
Realtime dashboard. Portfolio tracking. Basic indicators. Supabase auth.

### Q2 2026
Feature engineering. XGBoost classifier. Backtesting. Walk forward validation.

### Q3 2026
LSTM model. Ensemble XGBoost + LSTM. Realtime prediction API.

### Q4 2026
RL allocation agent. Model versioning. A/B testing.

### 2027+
Multi asset support. Sentiment data. Institutional risk tools. Third party API.

---

## Screenshots

### Home Dashboard
![Home Page](screenshots/home.png)
*Screenshot: Realtime market snapshot showing S&P 500, NASDAQ, DOW, VIX*

### Portfolio View
![Portfolio](screenshots/portfolio.png)
*Screenshot: User holdings with profit/loss and allocation pie chart*

### Asset Dashboard
![Asset Detail](screenshots/asset_detail.png)
*Screenshot: Individual stock with price chart and technical indicators*

### Volatility Regime
![Regime Analysis](screenshots/regime.png)
*Screenshot: ML features and regime classification display*

### Portfolio Tools
![Advanced Tools](screenshots/portfolio_tools.png)
*Screenshot: Efficient frontier and correlation matrix tools*

---

## File Structure

![Project Structure](docs/images/project_structure.png)

**Key Directories:**
- `app.py` - Main application entry point
- `frontend_v2/pages/` - All page modules (Home, AssetDashboard, VolatilityRegime, MyPortfolio, etc.)
- `frontend_v2/utils/` - Utility modules (auth.py, data_core.py)
- `frontend_v2/styles/` - Styling and theme components
- `src/` - Backend modules (features.py, data_fetch.py)
- `.streamlit/` - Configuration and secrets

---

## API Reference

**get_stock_data(ticker, period)**
Returns DataFrame with OHLCV. Cached 5 minutes.

**get_current_price(ticker)**
Returns current price as float. Cached 10 seconds.

**calculate_indicators(data)**
Returns DataFrame with RSI, MACD, Bollinger Bands. No cache.

**calculate_portfolio_metrics(portfolio_df)**
Returns dict with total value, P&L, Sharpe. No cache.

---

## Testing

### Unit Tests
```python
def test_calculate_rsi():
    data = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
    rsi = calculate_rsi(data, period=4)
    assert rsi.iloc[-1] > 50
```

### Integration Tests
Test yfinance connectivity. Test Supabase auth. Test chart rendering. Test portfolio operations.

---

**Last Updated:** December 2024
**Version:** 1.0.0

"""
Quantitative Market Augmentation Engine
=====================================

A comprehensive feature engineering pipeline that augments raw OHLCV data with:
1. Rolling returns, realized vol, EWMA vol
2. Tail risk metrics: VaR, ES (historical)
3. Structural regime labeling with volatility clustering (HMM or k-means)
4. Momentum factors: MA20, MA50, MA200, MACD, RSI
5. Liquidity indicators: VWAP, volume z-score, turnover ratio
6. Macro sensitivity tagging: beta to SPY, duration proxy via TLT correlation
7. Earnings impact flags with pre/post return deltas
8. Sentiment score embedding: VADER + finance-tuned LLM embedding average

Dependencies:
 - yfinance: Market data
 - pandas, numpy: Data manipulation
 - scipy: Statistical functions
 - sklearn: Clustering for regime detection
 - vaderSentiment: Sentiment analysis (optional)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

TRADING_DAYS_YEAR = 252
EWMA_LAMBDA = 0.94 # RiskMetrics standard decay factor
VIX_SYMBOL ="^VIX"
SPY_SYMBOL ="SPY"
TLT_SYMBOL ="TLT" # Duration/bond proxy

REGIME_THRESHOLDS = {
'low': 0.15, # VIX < 15 or realized vol < 15%
'medium': 0.20, # VIX 15-20
'high': 0.30, # VIX 20-30
'extreme': 0.30 # VIX > 30
}


# ============================================================
# 1. RETURNS & VOLATILITY CALCULATIONS
# ============================================================

def calculate_returns(prices: pd.Series, periods: List[int] = [1, 5, 21, 63]) -> pd.DataFrame:
"""
 Calculate rolling returns for multiple periods.
 
 Args:
 prices: Price series
 periods: List of lookback periods (1=daily, 5=weekly, 21=monthly, 63=quarterly)
 
 Returns:
 DataFrame with return columns for each period
"""
 returns = pd.DataFrame(index=prices.index)
 
 for period in periods:
 returns[f'return_{period}d'] = prices.pct_change(period)
 returns[f'log_return_{period}d'] = np.log(prices / prices.shift(period))
 
 return returns


def calculate_realized_volatility(prices: pd.Series, windows: List[int] = [10, 20, 60]) -> pd.DataFrame:
"""
 Calculate realized volatility using log returns (annualized).
 
 Args:
 prices: Price series
 windows: Rolling windows for volatility calculation
 
 Returns:
 DataFrame with realized volatility for each window
"""
 log_returns = np.log(prices / prices.shift(1))
 vol_df = pd.DataFrame(index=prices.index)
 
 for window in windows:
 vol_df[f'realized_vol_{window}d'] = (
 log_returns.rolling(window=window, min_periods=window // 2).std() 
 * np.sqrt(TRADING_DAYS_YEAR)
 )
 
 return vol_df


def calculate_ewma_volatility(prices: pd.Series, decay: float = EWMA_LAMBDA) -> pd.Series:
"""
 Calculate EWMA (Exponentially Weighted Moving Average) volatility.
 Uses RiskMetrics methodology with configurable decay factor.
 
 Args:
 prices: Price series
 decay: Decay factor (0.94 is RiskMetrics standard)
 
 Returns:
 Series of EWMA volatility estimates (annualized)
"""
 log_returns = np.log(prices / prices.shift(1))
 squared_returns = log_returns ** 2
 
 # EWMA variance
 ewma_var = squared_returns.ewm(alpha=1 - decay, adjust=False).mean()
 
 # Annualized volatility
 ewma_vol = np.sqrt(ewma_var * TRADING_DAYS_YEAR)
 
 return ewma_vol


def calculate_volatility_of_volatility(realized_vol: pd.Series, window: int = 20) -> pd.Series:
"""
 Calculate volatility-of-volatility (measures regime instability).

 Args:
 realized_vol: Realized volatility series
 window: Rolling window

 Returns:
 Vol-of-vol series
"""
 vol_returns = np.log(realized_vol / realized_vol.shift(1))
 vol_of_vol = vol_returns.rolling(window=window, min_periods=window // 2).std() * np.sqrt(TRADING_DAYS_YEAR)
 return vol_of_vol


# ============================================================
# 2. TAIL RISK METRICS: VaR & Expected Shortfall
# ============================================================

def calculate_var(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99],
 window: int = 252) -> pd.DataFrame:
"""
 Calculate rolling Historical VaR (Value at Risk).

 Args:
 returns: Return series
 confidence_levels: Confidence levels for VaR
 window: Rolling window for historical VaR

 Returns:
 DataFrame with VaR for each confidence level
"""
 var_df = pd.DataFrame(index=returns.index)

 for level in confidence_levels:
 pct = (1 - level) * 100
 var_df[f'VaR_{int(level*100)}'] = returns.rolling(window=window, min_periods=60).quantile(1 - level)

 return var_df


def calculate_expected_shortfall(returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99],
 window: int = 252) -> pd.DataFrame:
"""
 Calculate Expected Shortfall (Conditional VaR / CVaR).
 ES = average loss when VaR is exceeded.

 Args:
 returns: Return series
 confidence_levels: Confidence levels
 window: Rolling window

 Returns:
 DataFrame with ES for each confidence level
"""
 es_df = pd.DataFrame(index=returns.index)

 def rolling_es(x, level):
 var = np.percentile(x, (1 - level) * 100)
 return x[x <= var].mean() if len(x[x <= var]) > 0 else var

 for level in confidence_levels:
 es_df[f'ES_{int(level*100)}'] = returns.rolling(window=window, min_periods=60).apply(
 lambda x: rolling_es(x, level), raw=True
 )

 return es_df


def calculate_max_drawdown(prices: pd.Series, windows: List[int] = [60, 252]) -> pd.DataFrame:
"""
 Calculate rolling maximum drawdown.

 Args:
 prices: Price series
 windows: Rolling windows

 Returns:
 DataFrame with max drawdown for each window
"""
 dd_df = pd.DataFrame(index=prices.index)

 for window in windows:
 rolling_max = prices.rolling(window=window, min_periods=1).max()
 drawdown = (prices - rolling_max) / rolling_max
 dd_df[f'max_drawdown_{window}d'] = drawdown.rolling(window=window, min_periods=1).min()

 return dd_df


# ============================================================
# 3. REGIME LABELING (K-Means on Volatility Features)
# ============================================================

def classify_volatility_regime(vol_features: pd.DataFrame, n_regimes: int = 4) -> pd.DataFrame:
"""
 Classify volatility regimes using K-Means clustering on volatility features.

 Regimes: 0=Low, 1=Medium, 2=High, 3=Extreme

 Args:
 vol_features: DataFrame with volatility-related features
 n_regimes: Number of regimes to detect

 Returns:
 DataFrame with regime labels and probabilities
"""
 regime_df = pd.DataFrame(index=vol_features.index)

 # Select features for clustering
 feature_cols = [col for col in vol_features.columns if'vol' in col.lower()]
 if not feature_cols:
 feature_cols = vol_features.columns.tolist()

 # Clean data
 X = vol_features[feature_cols].dropna()

 if len(X) < n_regimes * 10:
 regime_df['regime_label'] = np.nan
 regime_df['regime_name'] ='Insufficient Data'
 return regime_df

 # Standardize features
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)

 # K-Means clustering
 kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
 labels = kmeans.fit_predict(X_scaled)

 # Create labeled series (aligned to original index)
 regime_series = pd.Series(index=vol_features.index, dtype=float)
 regime_series[X.index] = labels

 # Order regimes by average volatility (Low to Extreme)
 vol_means = X.iloc[:, 0].groupby(labels).mean()
 regime_order = vol_means.sort_values().index.tolist()
 regime_map = {old: new for new, old in enumerate(regime_order)}

 regime_df['regime_label'] = regime_series.map(regime_map)

 # Name regimes
 regime_names = {0:'Low', 1:'Medium', 2:'High', 3:'Extreme'}
 regime_df['regime_name'] = regime_df['regime_label'].map(regime_names)

 # Calculate regime persistence (days in current regime)
 regime_df['regime_persistence'] = (
 regime_df['regime_label'].groupby(
 (regime_df['regime_label'] != regime_df['regime_label'].shift()).cumsum()
 ).cumcount() + 1
 )

 return regime_df


# ============================================================
# 4. MOMENTUM FACTORS: MA, MACD, RSI
# ============================================================

def calculate_moving_averages(prices: pd.Series, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
"""
 Calculate simple and exponential moving averages.

 Args:
 prices: Price series
 windows: Moving average windows

 Returns:
 DataFrame with SMA, EMA, and price position relative to MAs
"""
 ma_df = pd.DataFrame(index=prices.index)

 for window in windows:
 ma_df[f'SMA_{window}'] = prices.rolling(window=window, min_periods=window // 2).mean()
 ma_df[f'EMA_{window}'] = prices.ewm(span=window, adjust=False).mean()
 ma_df[f'price_vs_SMA_{window}'] = (prices / ma_df[f'SMA_{window}'] - 1) * 100

 # Golden/Death cross signals
 if 50 in windows and 200 in windows:
 ma_df['golden_cross'] = (ma_df['SMA_50'] > ma_df['SMA_200']).astype(int)
 ma_df['death_cross'] = (ma_df['SMA_50'] < ma_df['SMA_200']).astype(int)

 return ma_df


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
"""
 Calculate MACD (Moving Average Convergence Divergence).

 Args:
 prices: Price series
 fast: Fast EMA period
 slow: Slow EMA period
 signal: Signal line period

 Returns:
 DataFrame with MACD, signal line, and histogram
"""
 macd_df = pd.DataFrame(index=prices.index)

 # Calculate EMAs
 ema_fast = prices.ewm(span=fast, adjust=False).mean()
 ema_slow = prices.ewm(span=slow, adjust=False).mean()

 # MACD line
 macd_df['MACD'] = ema_fast - ema_slow

 # Signal line
 macd_df['MACD_signal'] = macd_df['MACD'].ewm(span=signal, adjust=False).mean()

 # Histogram
 macd_df['MACD_histogram'] = macd_df['MACD'] - macd_df['MACD_signal']

 # Crossover signals
 macd_df['MACD_bullish'] = ((macd_df['MACD'] > macd_df['MACD_signal']) &
 (macd_df['MACD'].shift(1) <= macd_df['MACD_signal'].shift(1))).astype(int)
 macd_df['MACD_bearish'] = ((macd_df['MACD'] < macd_df['MACD_signal']) &
 (macd_df['MACD'].shift(1) >= macd_df['MACD_signal'].shift(1))).astype(int)

 return macd_df


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.DataFrame:
"""
 Calculate RSI (Relative Strength Index).

 Args:
 prices: Price series
 window: RSI calculation period

 Returns:
 DataFrame with RSI and overbought/oversold signals
"""
 rsi_df = pd.DataFrame(index=prices.index)

 # Calculate price changes
 delta = prices.diff()

 # Separate gains and losses
 gains = delta.where(delta > 0, 0)
 losses = -delta.where(delta < 0, 0)

 # Calculate average gains and losses (EMA)
 avg_gains = gains.ewm(span=window, adjust=False).mean()
 avg_losses = losses.ewm(span=window, adjust=False).mean()

 # Calculate RS and RSI
 rs = avg_gains / avg_losses
 rsi_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))

 # Overbought/Oversold signals
 rsi_df['RSI_overbought'] = (rsi_df[f'RSI_{window}'] > 70).astype(int)
 rsi_df['RSI_oversold'] = (rsi_df[f'RSI_{window}'] < 30).astype(int)

 return rsi_df


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
"""
 Calculate Bollinger Bands and band-derived indicators.

 Args:
 prices: Price series
 window: SMA window
 num_std: Number of standard deviations for bands

 Returns:
 DataFrame with bands, width, and %B
"""
 bb_df = pd.DataFrame(index=prices.index)

 # Middle band (SMA)
 bb_df['BB_middle'] = prices.rolling(window=window).mean()
 rolling_std = prices.rolling(window=window).std()

 # Upper and lower bands
 bb_df['BB_upper'] = bb_df['BB_middle'] + (rolling_std * num_std)
 bb_df['BB_lower'] = bb_df['BB_middle'] - (rolling_std * num_std)

 # Band width (volatility indicator)
 bb_df['BB_width'] = ((bb_df['BB_upper'] - bb_df['BB_lower']) / bb_df['BB_middle']) * 100

 # %B (position within bands)
 bb_df['BB_pct_b'] = (prices - bb_df['BB_lower']) / (bb_df['BB_upper'] - bb_df['BB_lower'])

 return bb_df


# ============================================================
# 5. LIQUIDITY INDICATORS: VWAP, Volume Z-Score, Turnover
# ============================================================

def calculate_vwap(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
"""
 Calculate Volume Weighted Average Price (VWAP).

 Args:
 ohlcv: OHLCV DataFrame
 window: Rolling window for VWAP

 Returns:
 VWAP series
"""
 typical_price = (ohlcv['High'] + ohlcv['Low'] + ohlcv['Close']) / 3
 vwap = (typical_price * ohlcv['Volume']).rolling(window).sum() / ohlcv['Volume'].rolling(window).sum()
 return vwap


def calculate_volume_indicators(volume: pd.Series, window: int = 20) -> pd.DataFrame:
"""
 Calculate volume-based indicators.

 Args:
 volume: Volume series
 window: Rolling window

 Returns:
 DataFrame with volume indicators
"""
 vol_df = pd.DataFrame(index=volume.index)

 # Volume moving average
 vol_df['volume_SMA'] = volume.rolling(window=window).mean()

 # Volume ratio (current vs average)
 vol_df['volume_ratio'] = volume / vol_df['volume_SMA']

 # Volume z-score
 vol_df['volume_zscore'] = (volume - vol_df['volume_SMA']) / volume.rolling(window=window).std()

 # Volume change
 vol_df['volume_change_pct'] = volume.pct_change() * 100

 # High volume days (>2 std above mean)
 vol_df['high_volume_day'] = (vol_df['volume_zscore'] > 2).astype(int)

 return vol_df


def calculate_turnover_ratio(volume: pd.Series, shares_outstanding: float = None,
 window: int = 20) -> pd.Series:
"""
 Calculate turnover ratio (volume / shares outstanding).
 If shares_outstanding not provided, uses average volume as proxy.

 Args:
 volume: Volume series
 shares_outstanding: Number of shares outstanding (optional)
 window: Rolling window

 Returns:
 Turnover ratio series
"""
 if shares_outstanding:
 return volume / shares_outstanding
 else:
 # Use rolling average as proxy
 avg_volume = volume.rolling(window=252, min_periods=60).mean()
 return volume / avg_volume


# ============================================================
# 6. MACRO SENSITIVITY: Beta to SPY, Duration Proxy (TLT Corr)
# ============================================================

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
"""
 Calculate rolling beta against market benchmark using correlation * vol ratio approach.

 Args:
 asset_returns: Asset return series
 market_returns: Market (SPY) return series
 window: Rolling window

 Returns:
 Rolling beta series
"""
 # Align series
 aligned = pd.DataFrame({'asset': asset_returns,'market': market_returns}).dropna()

 if len(aligned) < window:
 return pd.Series(index=asset_returns.index, dtype=float)

 # Use correlation * vol ratio approach (more stable)
 correlation = aligned['asset'].rolling(window, min_periods=30).corr(aligned['market'])
 asset_vol = aligned['asset'].rolling(window, min_periods=30).std()
 market_vol = aligned['market'].rolling(window, min_periods=30).std()

 beta = correlation * (asset_vol / market_vol)

 # Reindex to original asset_returns index
 return beta.reindex(asset_returns.index)


def calculate_duration_proxy(asset_returns: pd.Series, tlt_returns: pd.Series,
 window: int = 60) -> pd.DataFrame:
"""
 Calculate duration proxy using TLT correlation.
 Negative correlation with TLT suggests interest rate sensitivity.

 Args:
 asset_returns: Asset return series
 tlt_returns: TLT (bond ETF) return series
 window: Rolling window

 Returns:
 DataFrame with duration proxy metrics
"""
 dur_df = pd.DataFrame(index=asset_returns.index)

 # Rolling correlation with TLT
 dur_df['TLT_correlation'] = asset_returns.rolling(window).corr(tlt_returns)

 # Duration sensitivity estimate (beta to TLT)
 correlation = dur_df['TLT_correlation']
 asset_vol = asset_returns.rolling(window).std()
 tlt_vol = tlt_returns.rolling(window).std()

 dur_df['duration_beta'] = correlation * (asset_vol / tlt_vol)

 # Categorize sensitivity
 dur_df['rate_sensitive'] = (dur_df['TLT_correlation'].abs() > 0.3).astype(int)

 return dur_df


# ============================================================
# 7. EARNINGS IMPACT FLAGS
# ============================================================

def get_earnings_dates(symbol: str) -> pd.DataFrame:
"""
 Fetch earnings dates for a ticker.

 Args:
 symbol: Ticker symbol

 Returns:
 DataFrame with earnings dates
"""
 try:
 ticker = yf.Ticker(symbol)
 earnings = ticker.earnings_dates
 if earnings is not None and not earnings.empty:
 return earnings
 except Exception as e:
 logger.warning(f"Could not fetch earnings for {symbol}: {e}")

 return pd.DataFrame()


def calculate_earnings_impact(prices: pd.Series, earnings_dates: pd.DataFrame,
 pre_days: int = 5, post_days: int = 5) -> pd.DataFrame:
"""
 Calculate earnings impact metrics (pre/post return deltas).

 Args:
 prices: Price series
 earnings_dates: DataFrame with earnings dates
 pre_days: Days before earnings to measure
 post_days: Days after earnings to measure

 Returns:
 DataFrame with earnings impact flags and metrics
"""
 earn_df = pd.DataFrame(index=prices.index)
 earn_df['earnings_window'] = 0
 earn_df['pre_earnings_return'] = np.nan
 earn_df['post_earnings_return'] = np.nan
 earn_df['earnings_surprise_direction'] = 0

 if earnings_dates.empty:
 return earn_df

 returns = prices.pct_change()

 for date in earnings_dates.index:
 if date not in prices.index:
 # Find nearest trading day
 mask = prices.index <= date
 if mask.any():
 date = prices.index[mask][-1]
 else:
 continue

 try:
 date_loc = prices.index.get_loc(date)

 # Mark earnings window
 start_idx = max(0, date_loc - pre_days)
 end_idx = min(len(prices), date_loc + post_days + 1)
 earn_df.iloc[start_idx:end_idx, earn_df.columns.get_loc('earnings_window')] = 1

 # Pre-earnings return
 if date_loc >= pre_days:
 pre_return = (prices.iloc[date_loc] / prices.iloc[date_loc - pre_days] - 1)
 earn_df.iloc[date_loc, earn_df.columns.get_loc('pre_earnings_return')] = pre_return

 # Post-earnings return
 if date_loc + post_days < len(prices):
 post_return = (prices.iloc[date_loc + post_days] / prices.iloc[date_loc] - 1)
 earn_df.iloc[date_loc, earn_df.columns.get_loc('post_earnings_return')] = post_return

 # Surprise direction (positive or negative reaction)
 earn_df.iloc[date_loc, earn_df.columns.get_loc('earnings_surprise_direction')] = (
 1 if post_return > 0 else -1
 )

 except Exception:
 continue

 return earn_df


# ============================================================
# 8. SENTIMENT SCORING (VADER-based)
# ============================================================

def calculate_sentiment_score(headlines: List[str]) -> Dict[str, float]:
"""
 Calculate sentiment scores using VADER (or fallback to simple lexicon).

 Args:
 headlines: List of news headlines

 Returns:
 Dictionary with sentiment scores
"""
 try:
 from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 analyzer = SentimentIntensityAnalyzer()
 use_vader = True
 except ImportError:
 logger.warning("VADER not installed. Using simple sentiment scoring.")
 use_vader = False

 if not headlines:
 return {'compound': 0.0,'positive': 0.0,'negative': 0.0,'neutral': 0.0}

 if use_vader:
 scores = [analyzer.polarity_scores(h) for h in headlines]
 avg_scores = {
'compound': np.mean([s['compound'] for s in scores]),
'positive': np.mean([s['pos'] for s in scores]),
'negative': np.mean([s['neg'] for s in scores]),
'neutral': np.mean([s['neu'] for s in scores])
 }
 else:
 # Simple lexicon-based fallback
 positive_words = {'gain','rise','surge','jump','rally','up','high','growth','profit','beat'}
 negative_words = {'loss','fall','drop','crash','down','low','decline','miss','fear','sell'}

 pos_count = sum(1 for h in headlines for w in h.lower().split() if w in positive_words)
 neg_count = sum(1 for h in headlines for w in h.lower().split() if w in negative_words)
 total = pos_count + neg_count + 1

 avg_scores = {
'compound': (pos_count - neg_count) / total,
'positive': pos_count / total,
'negative': neg_count / total,
'neutral': 1 - (pos_count + neg_count) / total
 }

 return avg_scores


def create_sentiment_embedding(text: str, model_name: str ='all-MiniLM-L6-v2') -> np.ndarray:
"""
 Create embedding for text using sentence transformers (if available).
 Falls back to TF-IDF vectorization if not installed.

 Args:
 text: Text to embed
 model_name: Sentence transformer model name

 Returns:
 Embedding vector
"""
 try:
 from sentence_transformers import SentenceTransformer
 model = SentenceTransformer(model_name)
 return model.encode(text)
 except ImportError:
 logger.warning("sentence-transformers not installed. Returning zero embedding.")
 return np.zeros(384) # Default embedding size


# ============================================================
# MAIN AUGMENTATION ENGINE
# ============================================================

def fetch_benchmark_data(period: str ="2y") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
"""
 Fetch benchmark data for macro sensitivity calculations.

 Args:
 period: Data period

 Returns:
 Tuple of (SPY data, TLT data, VIX data)
"""
 try:
 spy = yf.Ticker(SPY_SYMBOL).history(period=period)
 tlt = yf.Ticker(TLT_SYMBOL).history(period=period)
 vix = yf.Ticker(VIX_SYMBOL).history(period=period)
 return spy, tlt, vix
 except Exception as e:
 logger.error(f"Error fetching benchmark data: {e}")
 return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def augment_asset_data(
 symbol: str,
 period: str ="2y",
 include_earnings: bool = True,
 include_sentiment: bool = False,
 headlines: List[str] = None
) -> pd.DataFrame:
"""
 Main augmentation function - creates comprehensive feature set from OHLCV data.

 Args:
 symbol: Ticker symbol
 period: Data period (1y, 2y, 5y, max)
 include_earnings: Whether to include earnings impact analysis
 include_sentiment: Whether to include sentiment scoring
 headlines: News headlines for sentiment analysis

 Returns:
 Augmented DataFrame with all engineered features
"""
 logger.info(f" Starting augmentation for {symbol}...")

 # Fetch main asset data
 try:
 ticker = yf.Ticker(symbol)
 data = ticker.history(period=period)

 if data.empty:
 logger.error(f"No data returned for {symbol}")
 return pd.DataFrame()

 except Exception as e:
 logger.error(f"Error fetching {symbol}: {e}")
 return pd.DataFrame()

 # Initialize augmented dataframe with original OHLCV
 augmented = data[['Open','High','Low','Close','Volume']].copy()

 # Fetch benchmark data
 spy_data, tlt_data, vix_data = fetch_benchmark_data(period)

 # ============================
 # 1. RETURNS & VOLATILITY
 # ============================
 logger.info(" Calculating returns and volatility...")

 # Rolling returns
 returns_df = calculate_returns(data['Close'], periods=[1, 5, 21, 63])
 augmented = pd.concat([augmented, returns_df], axis=1)

 # Realized volatility
 vol_df = calculate_realized_volatility(data['Close'], windows=[10, 20, 60])
 augmented = pd.concat([augmented, vol_df], axis=1)

 # EWMA volatility
 augmented['ewma_vol'] = calculate_ewma_volatility(data['Close'])

 # Vol-of-vol
 augmented['vol_of_vol'] = calculate_volatility_of_volatility(augmented['realized_vol_20d'])

 # ============================
 # 2. TAIL RISK METRICS
 # ============================
 logger.info(" Calculating tail risk metrics...")

 daily_returns = data['Close'].pct_change()

 # VaR
 var_df = calculate_var(daily_returns, confidence_levels=[0.95, 0.99])
 augmented = pd.concat([augmented, var_df], axis=1)

 # Expected Shortfall
 es_df = calculate_expected_shortfall(daily_returns, confidence_levels=[0.95, 0.99])
 augmented = pd.concat([augmented, es_df], axis=1)

 # Max Drawdown
 dd_df = calculate_max_drawdown(data['Close'], windows=[60, 252])
 augmented = pd.concat([augmented, dd_df], axis=1)

 # ============================
 # 3. REGIME LABELING
 # ============================
 logger.info(" Classifying volatility regimes...")

 vol_features = augmented[['realized_vol_10d','realized_vol_20d','ewma_vol']].copy()
 regime_df = classify_volatility_regime(vol_features)
 augmented = pd.concat([augmented, regime_df], axis=1)

 # ============================
 # 4. MOMENTUM FACTORS
 # ============================
 logger.info(" Calculating momentum factors...")

 # Moving averages
 ma_df = calculate_moving_averages(data['Close'], windows=[20, 50, 200])
 augmented = pd.concat([augmented, ma_df], axis=1)

 # MACD
 macd_df = calculate_macd(data['Close'])
 augmented = pd.concat([augmented, macd_df], axis=1)

 # RSI
 rsi_df = calculate_rsi(data['Close'])
 augmented = pd.concat([augmented, rsi_df], axis=1)

 # Bollinger Bands
 bb_df = calculate_bollinger_bands(data['Close'])
 augmented = pd.concat([augmented, bb_df], axis=1)

 # ============================
 # 5. LIQUIDITY INDICATORS
 # ============================
 logger.info(" Calculating liquidity indicators...")

 # VWAP
 augmented['VWAP'] = calculate_vwap(data)
 augmented['price_vs_VWAP'] = (data['Close'] / augmented['VWAP'] - 1) * 100

 # Volume indicators
 vol_ind_df = calculate_volume_indicators(data['Volume'])
 augmented = pd.concat([augmented, vol_ind_df], axis=1)

 # Turnover ratio
 augmented['turnover_ratio'] = calculate_turnover_ratio(data['Volume'])

 # ============================
 # 6. MACRO SENSITIVITY
 # ============================
 logger.info(" Calculating macro sensitivity...")

 if not spy_data.empty:
 spy_returns = spy_data['Close'].pct_change()
 asset_returns = data['Close'].pct_change()

 # Align data
 common_idx = asset_returns.index.intersection(spy_returns.index)
 if len(common_idx) > 60:
 augmented['beta_SPY'] = calculate_beta(
 asset_returns.loc[common_idx],
 spy_returns.loc[common_idx]
 )

 if not tlt_data.empty:
 tlt_returns = tlt_data['Close'].pct_change()
 asset_returns = data['Close'].pct_change()

 # Align data
 common_idx = asset_returns.index.intersection(tlt_returns.index)
 if len(common_idx) > 60:
 dur_df = calculate_duration_proxy(
 asset_returns.loc[common_idx],
 tlt_returns.loc[common_idx]
 )
 # Reindex to match augmented
 for col in dur_df.columns:
 augmented[col] = dur_df[col].reindex(augmented.index)

 # ============================
 # 7. EARNINGS IMPACT
 # ============================
 if include_earnings:
 logger.info(" Analyzing earnings impact...")
 earnings_dates = get_earnings_dates(symbol)
 earn_df = calculate_earnings_impact(data['Close'], earnings_dates)
 augmented = pd.concat([augmented, earn_df], axis=1)

 # ============================
 # 8. SENTIMENT SCORING
 # ============================
 if include_sentiment and headlines:
 logger.info(" Calculating sentiment scores...")
 sentiment = calculate_sentiment_score(headlines)
 augmented['sentiment_compound'] = sentiment['compound']
 augmented['sentiment_positive'] = sentiment['positive']
 augmented['sentiment_negative'] = sentiment['negative']

 # ============================
 # FINAL CLEANUP
 # ============================
 logger.info(" Cleaning up data...")

 # Add derived composite scores
 augmented['momentum_score'] = (
 augmented.get('RSI_14', 50) / 100 * 0.3 +
 augmented.get('MACD_histogram', 0).apply(lambda x: 0.5 + np.clip(x / 5, -0.5, 0.5)) * 0.3 +
 augmented.get('price_vs_SMA_50', 0).apply(lambda x: 0.5 + np.clip(x / 10, -0.5, 0.5)) * 0.4
 )

 # Trend strength score
 augmented['trend_strength'] = (
 (augmented['Close'] > augmented.get('SMA_20', augmented['Close'])).astype(int) +
 (augmented['Close'] > augmented.get('SMA_50', augmented['Close'])).astype(int) +
 (augmented['Close'] > augmented.get('SMA_200', augmented['Close'])).astype(int)
 ) / 3

 # Risk score (0-1, higher = riskier)
 if'realized_vol_20d' in augmented.columns and'VaR_95' in augmented.columns:
 vol_pct = augmented['realized_vol_20d'].rank(pct=True)
 var_pct = augmented['VaR_95'].abs().rank(pct=True)
 augmented['risk_score'] = (vol_pct + var_pct) / 2

 # Count features
 n_features = len(augmented.columns) - 5 # Exclude OHLCV
 logger.info(f" Augmentation complete! Added {n_features} engineered features.")

 return augmented


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_augmented_summary(augmented_df: pd.DataFrame) -> Dict:
"""
 Get summary statistics of augmented data.

 Args:
 augmented_df: Augmented DataFrame

 Returns:
 Summary dictionary
"""
 if augmented_df.empty:
 return {}

 latest = augmented_df.iloc[-1]

 summary = {
'symbol': augmented_df.index.name or'Unknown',
'date_range': {
'start': augmented_df.index.min().strftime('%Y-%m-%d'),
'end': augmented_df.index.max().strftime('%Y-%m-%d'),
'trading_days': len(augmented_df)
 },
'current_metrics': {
'price': latest.get('Close', np.nan),
'realized_vol_20d': latest.get('realized_vol_20d', np.nan),
'ewma_vol': latest.get('ewma_vol', np.nan),
'VaR_95': latest.get('VaR_95', np.nan),
'ES_95': latest.get('ES_95', np.nan),
'RSI': latest.get('RSI_14', np.nan),
'regime': latest.get('regime_name','Unknown'),
'beta': latest.get('beta_SPY', np.nan),
'momentum_score': latest.get('momentum_score', np.nan),
'risk_score': latest.get('risk_score', np.nan)
 },
'feature_count': len(augmented_df.columns),
'missing_pct': (augmented_df.isnull().sum().sum() / augmented_df.size) * 100
 }

 return summary


def export_augmented_data(augmented_df: pd.DataFrame, filepath: str, format: str ='parquet'):
"""
 Export augmented data to file.

 Args:
 augmented_df: Augmented DataFrame
 filepath: Output file path
 format: Output format ('parquet','csv','pickle')
"""
 if format =='parquet':
 augmented_df.to_parquet(filepath)
 elif format =='csv':
 augmented_df.to_csv(filepath)
 elif format =='pickle':
 augmented_df.to_pickle(filepath)
 else:
 raise ValueError(f"Unsupported format: {format}")

 logger.info(f" Exported augmented data to {filepath}")


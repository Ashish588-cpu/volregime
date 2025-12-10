# ================================
# features.py
# ================================
# Feature engineering module for volatility regime detection.
#
# This script takes the raw market data (SPY and VIX) and computes
# key technical indicators and risk metrics that help identify
# different market volatility regimes (low vol, high vol, crisis).
#
# Features computed:
#   1. Realized Volatility (10D, 20D) - measures actual price movement
#   2. Volatility-of-Volatility - measures how unstable volatility itself is
#   3. Rolling Drawdowns - measures peak-to-trough declines
#   4. SPY-TLT Correlation - flight-to-quality indicator (we'll use VIX as proxy)
#   5. RSI(14) - momentum oscillator
#   6. Bollinger Band Width(20D) - volatility bands indicator
#   7. ΔVIX (1D, 5D) - VIX changes as fear gauge
#
# Libraries:
#   pandas → data manipulation and rolling calculations
#   numpy → mathematical operations
#   pathlib → file path handling
# ================================

# ============================================================
# 1️⃣  Setup and Imports
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

# Get project root folder (volregime/)
ROOT = Path(__file__).resolve().parents[1]

# Define data directories
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "market.parquet"
OUTPUT_FILE = DATA_DIR / "features.parquet"


# ============================================================
# 2️⃣  Realized Volatility Functions
# ============================================================

def calculate_realized_volatility(prices, window=20):
    """
    Calculate realized volatility using log returns.

    Realized volatility measures the actual volatility that occurred
    in the market over a specific period. It's calculated as the
    standard deviation of log returns, annualized.

    Why we use this:
    - More accurate than implied volatility for historical analysis
    - Captures actual market stress vs. expected stress
    - Key input for volatility regime classification

    Args:
        prices (pd.Series): Price series (e.g., SPY adjusted close)
        window (int): Rolling window in days (10 or 20 typical)

    Returns:
        pd.Series: Annualized realized volatility
    """
    # Calculate log returns (more stable than simple returns)
    log_returns = np.log(prices / prices.shift(1))

    # Calculate rolling standard deviation
    rolling_std = log_returns.rolling(window=window, min_periods=window//2).std()

    # Annualize (multiply by sqrt of trading days per year)
    # 252 = typical trading days per year
    realized_vol = rolling_std * np.sqrt(252)

    return realized_vol


def calculate_volatility_of_volatility(realized_vol, window=20):
    """
    Calculate volatility-of-volatility (vol-of-vol).

    This measures how much the volatility itself is changing.
    High vol-of-vol indicates unstable, unpredictable markets
    where volatility is spiking and falling rapidly.

    Why we use this:
    - Identifies regime transitions (when vol becomes unstable)
    - Early warning of market stress
    - Distinguishes between stable high-vol and chaotic markets

    Args:
        realized_vol (pd.Series): Realized volatility series
        window (int): Rolling window for vol-of-vol calculation

    Returns:
        pd.Series: Volatility of volatility
    """
    # Calculate log returns of volatility
    vol_log_returns = np.log(realized_vol / realized_vol.shift(1))

    # Calculate rolling standard deviation of vol returns
    vol_of_vol = vol_log_returns.rolling(window=window, min_periods=window//2).std()

    # Annualize
    vol_of_vol = vol_of_vol * np.sqrt(252)

    return vol_of_vol


# ============================================================
# 3️⃣  Drawdown Functions
# ============================================================

def calculate_rolling_drawdown(prices, window=252):
    """
    Calculate rolling maximum drawdown.

    Drawdown measures the peak-to-trough decline in prices over
    a rolling window. It's a key risk metric that shows how much
    an investment has fallen from its recent peak.

    Why we use this:
    - Identifies periods of sustained market stress
    - Different from volatility (can have low vol but high drawdown)
    - Important for risk management and regime detection

    Args:
        prices (pd.Series): Price series
        window (int): Rolling window (252 = 1 year of trading days)

    Returns:
        pd.Series: Rolling maximum drawdown (negative values)
    """
    # Calculate rolling maximum (peak)
    rolling_max = prices.rolling(window=window, min_periods=1).max()

    # Calculate drawdown as percentage from peak
    drawdown = (prices - rolling_max) / rolling_max

    return drawdown


# ============================================================
# 4️⃣  RSI (Relative Strength Index) Function
# ============================================================

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and change
    of price movements. It oscillates between 0 and 100.
    - RSI > 70: Potentially overbought (sell signal)
    - RSI < 30: Potentially oversold (buy signal)

    Why we use this:
    - Identifies overbought/oversold conditions
    - Momentum indicator for regime changes
    - Complements volatility measures with trend information

    Args:
        prices (pd.Series): Price series
        window (int): Period for RSI calculation (14 is standard)

    Returns:
        pd.Series: RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(span=window, adjust=False).mean()
    avg_losses = losses.ewm(span=window, adjust=False).mean()

    # Calculate relative strength
    rs = avg_gains / avg_losses

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# 5️⃣  Bollinger Band Width Function
# ============================================================

def calculate_bollinger_band_width(prices, window=20, num_std=2):
    """
    Calculate Bollinger Band Width.

    Bollinger Bands consist of:
    - Middle line: Simple moving average
    - Upper band: SMA + (std_dev * num_std)
    - Lower band: SMA - (std_dev * num_std)

    Band width = (Upper Band - Lower Band) / Middle Line

    Why we use this:
    - Measures volatility in a normalized way
    - Wide bands = high volatility, narrow bands = low volatility
    - Helps identify volatility expansion/contraction cycles

    Args:
        prices (pd.Series): Price series
        window (int): Period for moving average and std dev (20 is standard)
        num_std (float): Number of standard deviations for bands (2 is standard)

    Returns:
        pd.Series: Bollinger Band Width (as percentage)
    """
    # Calculate simple moving average (middle line)
    sma = prices.rolling(window=window).mean()

    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=window).std()

    # Calculate upper and lower bands
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)

    # Calculate band width as percentage of middle line
    band_width = ((upper_band - lower_band) / sma) * 100

    return band_width


# ============================================================
# 6️⃣  VIX Change Functions
# ============================================================

def calculate_vix_changes(vix_prices):
    """
    Calculate VIX changes over different periods.

    VIX changes are a direct measure of fear/greed in the market.
    - Positive ΔVIX: Increasing fear/uncertainty
    - Negative ΔVIX: Decreasing fear/increasing complacency

    Why we use this:
    - VIX is the market's "fear gauge"
    - Rapid VIX changes signal regime transitions
    - Different timeframes capture different dynamics

    Args:
        vix_prices (pd.Series): VIX price series

    Returns:
        dict: Dictionary with 1D and 5D VIX changes
    """
    # 1-day VIX change (day-to-day fear changes)
    vix_change_1d = vix_prices.diff(1)

    # 5-day VIX change (weekly fear trend)
    vix_change_5d = vix_prices.diff(5)

    # calculate percentage changes for normalization
    vix_pct_change_1d = vix_prices.pct_change(1) * 100
    vix_pct_change_5d = vix_prices.pct_change(5) * 100

    return {
        'vix_change_1d': vix_change_1d,
        'vix_change_5d': vix_change_5d,
        'vix_pct_change_1d': vix_pct_change_1d,
        'vix_pct_change_5d': vix_pct_change_5d
    }


# ============================================================
# 7️⃣  Correlation Functions
# ============================================================

def calculate_rolling_correlation(series1, series2, window=60):
    """
    Calculate rolling correlation between two series.

    For volatility regime detection, we're interested in:
    - SPY-VIX correlation: Usually negative, but can break down in crisis
    - When correlation approaches zero or positive: Market stress signal

    Why we use this:
    - Identifies breakdown of normal market relationships
    - Flight-to-quality indicator
    - Regime change early warning system

    Args:
        series1 (pd.Series): First series (e.g., SPY returns)
        series2 (pd.Series): Second series (e.g., VIX returns)
        window (int): Rolling window for correlation (60 days = ~3 months)

    Returns:
        pd.Series: Rolling correlation coefficient (-1 to +1)
    """
    # Calculate returns for both series
    returns1 = series1.pct_change()
    returns2 = series2.pct_change()

    # Calculate rolling correlation
    rolling_corr = returns1.rolling(window=window).corr(returns2)

    return rolling_corr


# ============================================================
# 8️⃣  Main Feature Engineering Function
# ============================================================

def create_features(df):
    """
    Main function to create all features from raw market data.

    Takes the raw SPY and VIX data and computes all technical indicators
    and risk metrics needed for volatility regime detection.

    Args:
        df (pd.DataFrame): Raw market data with SPY and VIX columns

    Returns:
        pd.DataFrame: Feature dataset ready for modeling
    """
    print("🔧 Starting feature engineering...")

    # Extract price series (handle MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        spy_prices = df[('SPY', 'SPY')]
        vix_prices = df[('^VIX', '^VIX')]
    else:
        spy_prices = df['SPY']
        vix_prices = df['^VIX']

    # Initialize features dictionary
    features = {}

    # ============================================================
    # Feature 1: Realized Volatility (10D and 20D)
    # ============================================================
    print("📊 Computing realized volatility...")
    features['spy_realized_vol_10d'] = calculate_realized_volatility(spy_prices, window=10)
    features['spy_realized_vol_20d'] = calculate_realized_volatility(spy_prices, window=20)

    # ============================================================
    # Feature 2: Volatility-of-Volatility
    # ============================================================
    print("📈 Computing volatility-of-volatility...")
    features['spy_vol_of_vol'] = calculate_volatility_of_volatility(
        features['spy_realized_vol_20d'], window=20
    )

    # ============================================================
    # Feature 3: Rolling Drawdowns
    # ============================================================
    print("📉 Computing rolling drawdowns...")
    features['spy_drawdown_252d'] = calculate_rolling_drawdown(spy_prices, window=252)
    features['spy_drawdown_60d'] = calculate_rolling_drawdown(spy_prices, window=60)

    # ============================================================
    # Feature 4: SPY-VIX Correlation
    # ============================================================
    print("🔗 Computing SPY-VIX correlation...")
    features['spy_vix_correlation_60d'] = calculate_rolling_correlation(
        spy_prices, vix_prices, window=60
    )

    # ============================================================
    # Feature 5: RSI(14)
    # ============================================================
    print("⚡ Computing RSI...")
    features['spy_rsi_14'] = calculate_rsi(spy_prices, window=14)
    features['vix_rsi_14'] = calculate_rsi(vix_prices, window=14)

    # ============================================================
    # Feature 6: Bollinger Band Width(20D)
    # ============================================================
    print("📏 Computing Bollinger Band Width...")
    features['spy_bb_width_20d'] = calculate_bollinger_band_width(spy_prices, window=20)
    features['vix_bb_width_20d'] = calculate_bollinger_band_width(vix_prices, window=20)

    # ============================================================
    # Feature 7: ΔVIX (1D, 5D)
    # ============================================================
    print("😨 Computing VIX changes...")
    vix_changes = calculate_vix_changes(vix_prices)
    features.update(vix_changes)

    # ============================================================
    # Additional Features: Price levels and trends
    # ============================================================
    print("📊 Computing additional features...")

    # Price levels (normalized)
    features['spy_price'] = spy_prices
    features['vix_price'] = vix_prices

    # Simple moving averages for trend detection
    features['spy_sma_20'] = spy_prices.rolling(window=20).mean()
    features['spy_sma_50'] = spy_prices.rolling(window=50).mean()
    features['vix_sma_20'] = vix_prices.rolling(window=20).mean()

    # Price relative to moving averages (trend strength)
    features['spy_price_vs_sma20'] = (spy_prices / features['spy_sma_20'] - 1) * 100
    features['spy_price_vs_sma50'] = (spy_prices / features['spy_sma_50'] - 1) * 100
    features['vix_price_vs_sma20'] = (vix_prices / features['vix_sma_20'] - 1) * 100

    # ============================================================
    # Combine all features into DataFrame
    # ============================================================
    print("🔗 Combining features...")
    features_df = pd.DataFrame(features, index=df.index)

    # Add some basic feature interactions
    features_df['vol_regime_score'] = (
        features_df['spy_realized_vol_20d'] * 0.3 +
        features_df['spy_vol_of_vol'] * 0.2 +
        features_df['vix_pct_change_1d'].abs() * 0.2 +
        features_df['spy_bb_width_20d'] * 0.3
    )

    print(f"✅ Feature engineering complete! Created {len(features_df.columns)} features.")
    return features_df


# ============================================================
# 9️⃣  Data Quality and Validation Functions
# ============================================================

def validate_features(features_df):
    """
    Validate the computed features for data quality issues.

    Checks for:
    - Missing values
    - Infinite values
    - Extreme outliers
    - Feature distributions

    Args:
        features_df (pd.DataFrame): Features dataset

    Returns:
        dict: Validation report
    """
    print("🔍 Validating feature quality...")

    validation_report = {}

    # Check for missing values
    missing_counts = features_df.isnull().sum()
    validation_report['missing_values'] = missing_counts[missing_counts > 0]

    # Check for infinite values
    inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
    validation_report['infinite_values'] = inf_counts[inf_counts > 0]

    # Check for extreme outliers (beyond 5 standard deviations)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}

    for col in numeric_cols:
        if features_df[col].std() > 0:  # Avoid division by zero
            z_scores = np.abs((features_df[col] - features_df[col].mean()) / features_df[col].std())
            outlier_counts[col] = (z_scores > 5).sum()

    validation_report['extreme_outliers'] = {k: v for k, v in outlier_counts.items() if v > 0}

    # Basic statistics
    validation_report['feature_count'] = len(features_df.columns)
    validation_report['observation_count'] = len(features_df)
    validation_report['date_range'] = (features_df.index.min(), features_df.index.max())

    return validation_report


def print_validation_report(validation_report):
    """Print a formatted validation report."""
    print("\n" + "="*60)
    print("📋 FEATURE VALIDATION REPORT")
    print("="*60)

    print(f"📊 Dataset Overview:")
    print(f"   • Features: {validation_report['feature_count']}")
    print(f"   • Observations: {validation_report['observation_count']}")
    print(f"   • Date Range: {validation_report['date_range'][0]} to {validation_report['date_range'][1]}")

    if validation_report['missing_values'].empty:
        print(f"✅ No missing values found")
    else:
        print(f"⚠️  Missing values found:")
        for col, count in validation_report['missing_values'].items():
            print(f"   • {col}: {count} missing")

    if validation_report['infinite_values'].empty:
        print(f"✅ No infinite values found")
    else:
        print(f"❌ Infinite values found:")
        for col, count in validation_report['infinite_values'].items():
            print(f"   • {col}: {count} infinite")

    if not validation_report['extreme_outliers']:
        print(f"✅ No extreme outliers found")
    else:
        print(f"⚠️  Extreme outliers found (>5 std dev):")
        for col, count in validation_report['extreme_outliers'].items():
            print(f"   • {col}: {count} outliers")

    print("="*60)


# ============================================================
# 🔟  Main Execution Block
# ============================================================

def main():
    """
    Main execution function.

    1. Load raw market data
    2. Create features
    3. Validate features
    4. Save feature dataset
    """
    print(" Starting feature engineering pipeline...")
    print("="*60)

    # ============================================================
    # Step 1: Load raw market data
    # ============================================================
    try:
        print(f"📂 Loading market data from: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {INPUT_FILE}")
        print("💡 Make sure to run data_fetch.py first to create the market data file.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # ============================================================
    # Step 2: Create features
    # ============================================================
    try:
        features_df = create_features(df)
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        return

    # ============================================================
    # Step 3: Validate features
    # ============================================================
    try:
        validation_report = validate_features(features_df)
        print_validation_report(validation_report)
    except Exception as e:
        print(f"⚠️  Warning: Feature validation failed: {e}")

    # ============================================================
    # Step 4: Save feature dataset
    # ============================================================
    try:
        # Create output directory if it doesn't exist
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save features
        features_df.to_parquet(OUTPUT_FILE)
        print(f"\n💾 Features saved successfully to: {OUTPUT_FILE}")
        print(f"📊 Final dataset shape: {features_df.shape}")

        # Show sample of features          
        print(f"\n📋 Sample of computed features:")
        print(features_df.head().round(4))

        print("\n🎉 Feature engineering pipeline completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"❌ Error saving features: {e}")
        return


# ============================================================
# 🎯  Entry Point
# ============================================================

if __name__ == "__main__":
    main()
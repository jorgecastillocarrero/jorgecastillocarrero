"""
AI-powered financial analysis module.
Includes technical analysis, pattern detection, and optional OpenAI integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..config import get_settings
from ..database import get_db_manager

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis calculations for financial data."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = "close", periods: list[int] = None) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.

        Args:
            df: DataFrame with price data
            column: Column to use for calculation
            periods: List of periods (default: [20, 50, 200])

        Returns:
            DataFrame with SMA columns added
        """
        periods = periods or [20, 50, 200]
        result = df.copy()

        for period in periods:
            result[f"sma_{period}"] = result[column].rolling(window=period).mean()

        return result

    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = "close", periods: list[int] = None) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.

        Args:
            df: DataFrame with price data
            column: Column to use for calculation
            periods: List of periods (default: [12, 26])

        Returns:
            DataFrame with EMA columns added
        """
        periods = periods or [12, 26]
        result = df.copy()

        for period in periods:
            result[f"ema_{period}"] = result[column].ewm(span=period, adjust=False).mean()

        return result

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        Args:
            df: DataFrame with price data
            column: Column to use
            period: RSI period

        Returns:
            DataFrame with RSI column added
        """
        result = df.copy()
        delta = result[column].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        result["rsi"] = 100 - (100 / (1 + rs))

        return result

    @staticmethod
    def calculate_macd(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with price data
            column: Column to use

        Returns:
            DataFrame with MACD columns added
        """
        result = df.copy()

        ema_12 = result[column].ewm(span=12, adjust=False).mean()
        ema_26 = result[column].ewm(span=26, adjust=False).mean()

        result["macd"] = ema_12 - ema_26
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        return result

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = "close", period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data
            column: Column to use
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            DataFrame with Bollinger Band columns added
        """
        result = df.copy()

        sma = result[column].rolling(window=period).mean()
        std = result[column].rolling(window=period).std()

        result["bb_upper"] = sma + (std * std_dev)
        result["bb_middle"] = sma
        result["bb_lower"] = sma - (std * std_dev)

        return result

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            DataFrame with ATR column added
        """
        result = df.copy()

        high_low = result["high"] - result["low"]
        high_close = abs(result["high"] - result["close"].shift())
        low_close = abs(result["low"] - result["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["atr"] = true_range.rolling(window=period).mean()

        return result


class PatternDetector:
    """Detect common chart patterns."""

    @staticmethod
    def detect_trend(df: pd.DataFrame, period: int = 20) -> str:
        """
        Detect current trend direction.

        Args:
            df: DataFrame with price data
            period: Period for trend detection

        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        if len(df) < period:
            return "insufficient_data"

        recent = df.tail(period)
        price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

        if price_change > 0.05:
            return "uptrend"
        elif price_change < -0.05:
            return "downtrend"
        else:
            return "sideways"

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
        """
        Detect support and resistance levels.

        Args:
            df: DataFrame with price data
            window: Window for level detection

        Returns:
            Dictionary with support and resistance levels
        """
        if len(df) < window:
            return {"support": None, "resistance": None}

        recent = df.tail(window * 3) if len(df) >= window * 3 else df

        highs = recent["high"].rolling(window=window, center=True).max()
        lows = recent["low"].rolling(window=window, center=True).min()

        resistance = highs.dropna().mode()
        support = lows.dropna().mode()

        return {
            "support": support.iloc[0] if len(support) > 0 else recent["low"].min(),
            "resistance": resistance.iloc[0] if len(resistance) > 0 else recent["high"].max(),
        }

    @staticmethod
    def detect_crossovers(df: pd.DataFrame) -> list[dict]:
        """
        Detect moving average crossovers.

        Args:
            df: DataFrame with SMA columns

        Returns:
            List of crossover events
        """
        crossovers = []

        if "sma_20" not in df.columns or "sma_50" not in df.columns:
            return crossovers

        for i in range(1, len(df)):
            # Golden cross (20 crosses above 50)
            if (
                df["sma_20"].iloc[i - 1] < df["sma_50"].iloc[i - 1]
                and df["sma_20"].iloc[i] > df["sma_50"].iloc[i]
            ):
                crossovers.append(
                    {
                        "date": df.index[i],
                        "type": "golden_cross",
                        "signal": "bullish",
                    }
                )

            # Death cross (20 crosses below 50)
            if (
                df["sma_20"].iloc[i - 1] > df["sma_50"].iloc[i - 1]
                and df["sma_20"].iloc[i] < df["sma_50"].iloc[i]
            ):
                crossovers.append(
                    {
                        "date": df.index[i],
                        "type": "death_cross",
                        "signal": "bearish",
                    }
                )

        return crossovers


class TrendPredictor:
    """Simple trend prediction using linear regression."""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def predict_trend(self, df: pd.DataFrame, forecast_days: int = 5) -> dict:
        """
        Predict price trend using linear regression.

        Args:
            df: DataFrame with price data
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with prediction results
        """
        if len(df) < 30:
            return {"error": "Insufficient data for prediction"}

        # Prepare features
        recent = df.tail(60).copy()
        recent["day_num"] = range(len(recent))

        X = recent[["day_num"]].values
        y = recent["close"].values

        # Fit model
        self.model.fit(X, y)

        # Predict
        future_days = np.array([[len(recent) + i] for i in range(forecast_days)])
        predictions = self.model.predict(future_days)

        # Calculate metrics
        current_price = recent["close"].iloc[-1]
        predicted_price = predictions[-1]
        change_percent = ((predicted_price - current_price) / current_price) * 100

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_percent": change_percent,
            "forecast_days": forecast_days,
            "trend": "bullish" if change_percent > 0 else "bearish",
            "predictions": list(predictions),
            "r_squared": self.model.score(X, y),
        }


class AIAnalyzer:
    """Main analyzer class combining all analysis methods."""

    def __init__(self):
        self.settings = get_settings()
        self.db = get_db_manager()
        self.technical = TechnicalAnalyzer()
        self.patterns = PatternDetector()
        self.predictor = TrendPredictor()
        self._openai_client = None

    def _get_openai_client(self):
        """Get OpenAI client (lazy initialization)."""
        if self._openai_client is None and self.settings.is_openai_configured:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def analyze_symbol(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Perform complete analysis on a symbol.

        Args:
            df: DataFrame with price history
            symbol: Symbol being analyzed

        Returns:
            Complete analysis results
        """
        if df.empty:
            return {"error": "No data available"}

        # Calculate technical indicators
        df = self.technical.calculate_sma(df)
        df = self.technical.calculate_ema(df)
        df = self.technical.calculate_rsi(df)
        df = self.technical.calculate_macd(df)
        df = self.technical.calculate_bollinger_bands(df)
        df = self.technical.calculate_atr(df)

        # Get latest values
        latest = df.iloc[-1]

        # Detect patterns
        trend = self.patterns.detect_trend(df)
        levels = self.patterns.detect_support_resistance(df)
        crossovers = self.patterns.detect_crossovers(df)

        # Get prediction
        prediction = self.predictor.predict_trend(df)

        # Calculate returns
        returns_1d = ((latest["close"] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100) if len(df) > 1 else 0
        returns_7d = ((latest["close"] - df["close"].iloc[-8]) / df["close"].iloc[-8] * 100) if len(df) > 7 else 0
        returns_30d = ((latest["close"] - df["close"].iloc[-31]) / df["close"].iloc[-31] * 100) if len(df) > 30 else 0

        return {
            "symbol": symbol,
            "analysis_date": datetime.utcnow().isoformat(),
            "current_price": latest["close"],
            "volume": latest.get("volume", 0),
            "technical_indicators": {
                "sma_20": latest.get("sma_20"),
                "sma_50": latest.get("sma_50"),
                "sma_200": latest.get("sma_200"),
                "ema_12": latest.get("ema_12"),
                "ema_26": latest.get("ema_26"),
                "rsi": latest.get("rsi"),
                "macd": latest.get("macd"),
                "macd_signal": latest.get("macd_signal"),
                "bb_upper": latest.get("bb_upper"),
                "bb_lower": latest.get("bb_lower"),
                "atr": latest.get("atr"),
            },
            "trend": trend,
            "support_resistance": levels,
            "recent_crossovers": crossovers[-5:] if crossovers else [],
            "prediction": prediction,
            "returns": {
                "1d": returns_1d,
                "7d": returns_7d,
                "30d": returns_30d,
            },
            "signals": self._generate_signals(latest, trend),
        }

    def _generate_signals(self, latest: pd.Series, trend: str) -> list[dict]:
        """Generate trading signals based on indicators."""
        signals = []

        # RSI signals
        rsi = latest.get("rsi")
        if rsi is not None:
            if rsi < 30:
                signals.append({"indicator": "RSI", "signal": "oversold", "action": "potential_buy"})
            elif rsi > 70:
                signals.append({"indicator": "RSI", "signal": "overbought", "action": "potential_sell"})

        # MACD signals
        macd = latest.get("macd")
        macd_signal = latest.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals.append({"indicator": "MACD", "signal": "bullish", "action": "hold_buy"})
            else:
                signals.append({"indicator": "MACD", "signal": "bearish", "action": "hold_sell"})

        # Trend signal
        signals.append({"indicator": "Trend", "signal": trend, "action": trend})

        return signals

    def get_ai_summary(self, analysis: dict) -> str | None:
        """
        Get AI-generated summary of analysis (requires OpenAI).

        Args:
            analysis: Analysis results dictionary

        Returns:
            AI-generated summary or None if OpenAI not configured
        """
        client = self._get_openai_client()
        if client is None:
            return None

        prompt = f"""Analyze this financial data and provide a brief summary:

Symbol: {analysis['symbol']}
Current Price: ${analysis['current_price']:.2f}
Trend: {analysis['trend']}
RSI: {analysis['technical_indicators'].get('rsi', 'N/A'):.1f}
Returns (1d/7d/30d): {analysis['returns']['1d']:.2f}% / {analysis['returns']['7d']:.2f}% / {analysis['returns']['30d']:.2f}%

Signals: {analysis['signals']}

Provide a 2-3 sentence summary of the current market situation and potential outlook.
"""

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Be concise and objective."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None


# CLI test functionality
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing AI Analyzer ===\n")

    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(100),
            "high": prices + abs(np.random.randn(100)),
            "low": prices - abs(np.random.randn(100)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    analyzer = AIAnalyzer()
    results = analyzer.analyze_symbol(df, "TEST.US")

    print(f"Symbol: {results['symbol']}")
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Trend: {results['trend']}")
    print(f"RSI: {results['technical_indicators']['rsi']:.2f}")
    print(f"Support: ${results['support_resistance']['support']:.2f}")
    print(f"Resistance: ${results['support_resistance']['resistance']:.2f}")
    print(f"\nSignals:")
    for signal in results["signals"]:
        print(f"  - {signal['indicator']}: {signal['signal']} ({signal['action']})")

    print(f"\nPrediction (5 days): {results['prediction']['trend']}")
    print(f"R-squared: {results['prediction']['r_squared']:.3f}")

    print("\n=== AI Analyzer test completed ===")

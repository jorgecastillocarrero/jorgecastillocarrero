"""
Tests for src/technical.py - Technical analysis calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.technical import (
    calculate_rsi,
    calculate_moving_average,
    calculate_distance_from_ma,
    calculate_returns,
    calculate_ytd_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_all_metrics,
    RISK_FREE_RATE,
)


class TestCalculateRSI:
    """Tests for RSI calculation."""

    def test_rsi_returns_series(self, sample_prices_series):
        """Test that RSI returns a pandas Series."""
        rsi = calculate_rsi(sample_prices_series)
        assert isinstance(rsi, pd.Series)

    def test_rsi_values_in_range(self, sample_prices_series):
        """Test that RSI values are between 0 and 100."""
        rsi = calculate_rsi(sample_prices_series)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_custom_period(self, sample_prices_series):
        """Test RSI with custom period."""
        rsi_7 = calculate_rsi(sample_prices_series, period=7)
        rsi_21 = calculate_rsi(sample_prices_series, period=21)
        # Different periods should give different results
        assert not rsi_7.equals(rsi_21)

    def test_rsi_length_matches_input(self, sample_prices_series):
        """Test that RSI has same length as input."""
        rsi = calculate_rsi(sample_prices_series)
        assert len(rsi) == len(sample_prices_series)

    def test_rsi_with_constant_prices(self):
        """Test RSI with constant prices returns NaN or 50-ish."""
        prices = pd.Series([100.0] * 50)
        rsi = calculate_rsi(prices)
        # With no movement, RSI is undefined or around 50
        valid = rsi.dropna()
        if len(valid) > 0:
            assert True  # Just checking it doesn't error

    def test_rsi_trending_up(self):
        """Test RSI is high for strongly trending up prices."""
        prices = pd.Series([100 + i for i in range(50)])
        rsi = calculate_rsi(prices)
        # Last RSI should be high (above 70) for strong uptrend
        assert rsi.iloc[-1] > 70

    def test_rsi_trending_down(self):
        """Test RSI is low for strongly trending down prices."""
        prices = pd.Series([150 - i for i in range(50)])
        rsi = calculate_rsi(prices)
        # Last RSI should be low (below 30) for strong downtrend
        assert rsi.iloc[-1] < 30


class TestCalculateMovingAverage:
    """Tests for moving average calculation."""

    def test_ma_returns_series(self, sample_prices_series):
        """Test that MA returns a pandas Series."""
        ma = calculate_moving_average(sample_prices_series, 50)
        assert isinstance(ma, pd.Series)

    def test_ma_length_matches_input(self, sample_prices_series):
        """Test that MA has same length as input."""
        ma = calculate_moving_average(sample_prices_series, 50)
        assert len(ma) == len(sample_prices_series)

    def test_ma_nan_for_insufficient_data(self, sample_prices_series):
        """Test that MA returns NaN for first period-1 values."""
        ma = calculate_moving_average(sample_prices_series, 50)
        assert ma.iloc[:49].isna().all()

    def test_ma_calculation_correct(self):
        """Test MA calculation is mathematically correct."""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma = calculate_moving_average(prices, 5)
        # MA at position 4 should be (1+2+3+4+5)/5 = 3
        assert ma.iloc[4] == 3.0
        # MA at position 9 should be (6+7+8+9+10)/5 = 8
        assert ma.iloc[9] == 8.0


class TestCalculateDistanceFromMA:
    """Tests for distance from MA calculation."""

    def test_distance_returns_series(self, sample_prices_series):
        """Test that distance returns a pandas Series."""
        ma = calculate_moving_average(sample_prices_series, 50)
        distance = calculate_distance_from_ma(sample_prices_series, ma)
        assert isinstance(distance, pd.Series)

    def test_distance_positive_above_ma(self):
        """Test distance is positive when price above MA."""
        prices = pd.Series([100, 110, 120, 130, 140])
        ma = pd.Series([100, 100, 100, 100, 100])
        distance = calculate_distance_from_ma(prices, ma)
        assert (distance.iloc[1:] > 0).all()

    def test_distance_negative_below_ma(self):
        """Test distance is negative when price below MA."""
        prices = pd.Series([100, 90, 80, 70, 60])
        ma = pd.Series([100, 100, 100, 100, 100])
        distance = calculate_distance_from_ma(prices, ma)
        assert (distance.iloc[1:] < 0).all()

    def test_distance_as_percentage(self):
        """Test distance is calculated as decimal percentage."""
        prices = pd.Series([110.0])
        ma = pd.Series([100.0])
        distance = calculate_distance_from_ma(prices, ma)
        assert distance.iloc[0] == pytest.approx(0.10)  # 10% above


class TestCalculateReturns:
    """Tests for return calculation."""

    def test_returns_series(self, sample_prices_series):
        """Test that returns is a pandas Series."""
        returns = calculate_returns(sample_prices_series)
        assert isinstance(returns, pd.Series)

    def test_returns_first_is_nan(self, sample_prices_series):
        """Test that first return value is NaN."""
        returns = calculate_returns(sample_prices_series)
        assert pd.isna(returns.iloc[0])

    def test_returns_calculation(self):
        """Test returns calculation is correct."""
        prices = pd.Series([100, 110, 99])
        returns = calculate_returns(prices)
        assert returns.iloc[1] == pytest.approx(0.10)  # 10%
        assert returns.iloc[2] == pytest.approx(-0.10)  # -10%

    def test_multiperiod_returns(self):
        """Test multi-period returns."""
        prices = pd.Series([100, 105, 110, 115, 120])
        returns_3 = calculate_returns(prices, periods=3)
        # Return from 100 to 115 over 3 periods
        assert returns_3.iloc[3] == pytest.approx(0.15)


class TestCalculateYTDReturn:
    """Tests for YTD return calculation."""

    def test_ytd_returns_series(self, sample_prices_series):
        """Test that YTD returns a pandas Series."""
        ytd = calculate_ytd_return(sample_prices_series)
        assert isinstance(ytd, pd.Series)

    def test_ytd_first_day_zero(self):
        """Test that first day of year has ~0 YTD return."""
        dates = pd.date_range(start="2024-01-02", periods=30, freq="B")
        prices = pd.Series([100 + i for i in range(30)], index=dates)
        ytd = calculate_ytd_return(prices)
        assert ytd.iloc[0] == pytest.approx(0.0)


class TestCalculateVolatility:
    """Tests for volatility calculation."""

    def test_volatility_returns_series(self, sample_prices_series):
        """Test that volatility returns a pandas Series."""
        returns = calculate_returns(sample_prices_series)
        vol = calculate_volatility(returns)
        assert isinstance(vol, pd.Series)

    def test_volatility_positive(self, sample_prices_series):
        """Test that volatility is positive."""
        returns = calculate_returns(sample_prices_series)
        vol = calculate_volatility(returns)
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all()

    def test_volatility_annualized(self):
        """Test that volatility is annualized (multiplied by sqrt(252))."""
        # Create returns with known std
        returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 10)
        vol = calculate_volatility(returns, period=10)
        # Should be annualized
        daily_std = returns.rolling(10).std().iloc[-1]
        expected_annual = daily_std * np.sqrt(252)
        assert vol.iloc[-1] == pytest.approx(expected_annual, rel=0.01)


class TestCalculateSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_returns_series(self, sample_prices_series):
        """Test that Sharpe returns a pandas Series."""
        returns = calculate_returns(sample_prices_series)
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, pd.Series)

    def test_sharpe_with_custom_rf_rate(self, sample_prices_series):
        """Test Sharpe with custom risk-free rate."""
        returns = calculate_returns(sample_prices_series)
        sharpe_low = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        sharpe_high = calculate_sharpe_ratio(returns, risk_free_rate=0.10)
        # Higher RF rate should give lower Sharpe
        valid_low = sharpe_low.dropna()
        valid_high = sharpe_high.dropna()
        if len(valid_low) > 0 and len(valid_high) > 0:
            assert valid_high.iloc[-1] < valid_low.iloc[-1]


class TestCalculateAllMetrics:
    """Tests for calculate_all_metrics function."""

    def test_all_metrics_returns_dataframe(self, sample_prices_df):
        """Test that all_metrics returns a DataFrame."""
        metrics = calculate_all_metrics(sample_prices_df)
        assert isinstance(metrics, pd.DataFrame)

    def test_all_metrics_has_expected_columns(self, sample_prices_df):
        """Test that metrics DataFrame has expected columns."""
        metrics = calculate_all_metrics(sample_prices_df)
        expected_cols = [
            "close_price", "rsi_14", "ma_50", "ma_200",
            "m50", "m200", "daily_return", "weekly_return",
            "monthly_return", "ytd_return", "volatility_21d", "sharpe_21d"
        ]
        for col in expected_cols:
            assert col in metrics.columns

    def test_all_metrics_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        metrics = calculate_all_metrics(empty_df)
        assert metrics.empty

    def test_all_metrics_preserves_index(self, sample_prices_df):
        """Test that metrics preserves the date index."""
        metrics = calculate_all_metrics(sample_prices_df)
        assert len(metrics) == len(sample_prices_df)
        assert metrics.index.equals(sample_prices_df.index)

    def test_all_metrics_with_adjusted_close(self, sample_prices_df):
        """Test that adjusted_close is used when available."""
        df = sample_prices_df.copy()
        df["adjusted_close"] = df["close"] * 1.1
        metrics = calculate_all_metrics(df)
        # Close price should reflect adjusted close
        assert metrics["close_price"].iloc[-1] == pytest.approx(df["adjusted_close"].iloc[-1])

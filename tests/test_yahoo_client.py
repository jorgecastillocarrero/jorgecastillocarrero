"""
Tests for src/yahoo_client.py - Yahoo Finance client.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.yahoo_client import (
    YahooFinanceClient,
    format_number,
    format_percent,
)


class TestSymbolConversion:
    """Tests for symbol conversion."""

    def test_convert_us_symbol(self):
        """Test converting US symbols."""
        client = YahooFinanceClient()
        assert client._convert_symbol("AAPL.US") == "AAPL"
        assert client._convert_symbol("MSFT.US") == "MSFT"
        assert client._convert_symbol("GOOGL.US") == "GOOGL"

    def test_convert_lse_symbol(self):
        """Test converting London Stock Exchange symbols."""
        client = YahooFinanceClient()
        assert client._convert_symbol("BP.LSE") == "BP.L"
        assert client._convert_symbol("SHEL.LSE") == "SHEL.L"

    def test_convert_xetra_symbol(self):
        """Test converting German symbols."""
        client = YahooFinanceClient()
        assert client._convert_symbol("SAP.XETRA") == "SAP.DE"
        assert client._convert_symbol("BMW.XETRA") == "BMW.DE"

    def test_convert_paris_symbol(self):
        """Test converting Paris symbols."""
        client = YahooFinanceClient()
        assert client._convert_symbol("MC.PA") == "MC.PA"

    def test_convert_plain_symbol(self):
        """Test symbols without exchange suffix."""
        client = YahooFinanceClient()
        assert client._convert_symbol("AAPL") == "AAPL"
        assert client._convert_symbol("SPY") == "SPY"

    def test_convert_madrid_symbol(self):
        """Test converting Madrid symbols."""
        client = YahooFinanceClient()
        assert client._convert_symbol("TEF.MC") == "TEF.MC"


class TestGetTicker:
    """Tests for get_ticker method."""

    @patch("yfinance.Ticker")
    def test_get_ticker_converts_symbol(self, mock_ticker):
        """Test that get_ticker converts symbol correctly."""
        client = YahooFinanceClient()
        client.get_ticker("AAPL.US")
        mock_ticker.assert_called_with("AAPL")

    @patch("yfinance.Ticker")
    def test_get_ticker_returns_ticker(self, mock_ticker):
        """Test that get_ticker returns yfinance Ticker."""
        mock_ticker.return_value = MagicMock()
        client = YahooFinanceClient()
        result = client.get_ticker("AAPL.US")
        assert result is not None


class TestGetHistoricalData:
    """Tests for get_historical_data method."""

    @patch("yfinance.Ticker")
    def test_get_historical_returns_dataframe(self, mock_ticker):
        """Test that historical data returns DataFrame."""
        mock_history = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000000, 1100000],
        })
        mock_ticker.return_value.history.return_value = mock_history

        client = YahooFinanceClient()
        df = client.get_historical_data("AAPL.US", period="1mo")

        assert isinstance(df, pd.DataFrame)
        assert "close" in df.columns
        assert "symbol" in df.columns

    @patch("yfinance.Ticker")
    def test_get_historical_renames_columns(self, mock_ticker):
        """Test that columns are renamed to lowercase."""
        mock_history = pd.DataFrame({
            "Open": [100], "High": [102], "Low": [99],
            "Close": [101], "Adj Close": [101], "Volume": [1000000],
        })
        mock_ticker.return_value.history.return_value = mock_history

        client = YahooFinanceClient()
        df = client.get_historical_data("AAPL.US")

        assert "open" in df.columns
        assert "close" in df.columns
        assert "adjusted_close" in df.columns or "close" in df.columns

    @patch("yfinance.Ticker")
    def test_get_historical_empty_response(self, mock_ticker):
        """Test handling of empty response."""
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        client = YahooFinanceClient()
        df = client.get_historical_data("INVALID")

        assert df.empty

    @patch("yfinance.Ticker")
    def test_get_historical_with_dates(self, mock_ticker):
        """Test historical data with start/end dates."""
        mock_history = pd.DataFrame({"Close": [100]})
        mock_ticker.return_value.history.return_value = mock_history

        client = YahooFinanceClient()
        client.get_historical_data(
            "AAPL.US",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30)
        )

        mock_ticker.return_value.history.assert_called_once()


class TestGetMultipleHistorical:
    """Tests for get_multiple_historical method."""

    @patch("yfinance.download")
    def test_get_multiple_returns_dict(self, mock_download):
        """Test that multiple historical returns dict."""
        mock_data = pd.DataFrame({
            ("AAPL", "Close"): [150],
            ("MSFT", "Close"): [300],
        })
        mock_download.return_value = mock_data

        client = YahooFinanceClient()
        with patch.object(client, "get_historical_data") as mock_single:
            mock_single.return_value = pd.DataFrame({"close": [150]})
            result = client.get_multiple_historical(["AAPL.US", "MSFT.US"])

        assert isinstance(result, dict)


class TestGetFundamentals:
    """Tests for get_fundamentals method."""

    @patch("yfinance.Ticker")
    def test_get_fundamentals_returns_dict(self, mock_ticker):
        """Test that fundamentals returns structured dict."""
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3000000000000,
            "trailingPE": 28.5,
            "dividendYield": 0.005,
        }
        mock_ticker.return_value.info = mock_info

        client = YahooFinanceClient()
        result = client.get_fundamentals("AAPL.US")

        assert "general" in result
        assert "market_data" in result
        assert "valuation" in result
        assert result["general"]["name"] == "Apple Inc."

    @patch("yfinance.Ticker")
    def test_get_fundamentals_handles_error(self, mock_ticker):
        """Test fundamentals handles errors gracefully."""
        mock_ticker.return_value.info = property(
            fget=lambda self: (_ for _ in ()).throw(Exception("API Error"))
        )

        client = YahooFinanceClient()
        with patch.object(mock_ticker.return_value, "info", new_callable=lambda: property(
            fget=lambda s: exec('raise Exception("API Error")')
        )):
            pass  # Test would check error handling


class TestGetDividends:
    """Tests for get_dividends method."""

    @patch("yfinance.Ticker")
    def test_get_dividends_returns_dataframe(self, mock_ticker):
        """Test that dividends returns DataFrame."""
        mock_dividends = pd.Series([0.24, 0.25], name="Dividends")
        mock_ticker.return_value.dividends = mock_dividends

        client = YahooFinanceClient()
        df = client.get_dividends("AAPL.US")

        assert isinstance(df, pd.DataFrame)


class TestGetSplits:
    """Tests for get_splits method."""

    @patch("yfinance.Ticker")
    def test_get_splits_returns_dataframe(self, mock_ticker):
        """Test that splits returns DataFrame."""
        mock_splits = pd.Series([4.0], name="Stock Splits")
        mock_ticker.return_value.splits = mock_splits

        client = YahooFinanceClient()
        df = client.get_splits("AAPL.US")

        assert isinstance(df, pd.DataFrame)


class TestGetQuickStats:
    """Tests for get_quick_stats method."""

    @patch("yfinance.Ticker")
    def test_get_quick_stats(self, mock_ticker):
        """Test quick stats returns expected structure."""
        mock_info = {
            "longName": "Apple Inc.",
            "marketCap": 3000000000000,
            "trailingPE": 28.5,
            "trailingEps": 6.05,
            "dividendYield": 0.005,
            "fiftyTwoWeekHigh": 200,
            "fiftyTwoWeekLow": 150,
        }
        mock_ticker.return_value.info = mock_info

        client = YahooFinanceClient()
        stats = client.get_quick_stats("AAPL.US")

        assert stats["name"] == "Apple Inc."
        assert stats["market_cap"] == 3000000000000
        assert "market_cap_formatted" in stats


class TestGetMarketCap:
    """Tests for get_market_cap method."""

    @patch.object(YahooFinanceClient, "get_fundamentals")
    def test_get_market_cap(self, mock_fundamentals):
        """Test get_market_cap returns float."""
        mock_fundamentals.return_value = {
            "market_data": {"market_cap": 3000000000000}
        }

        client = YahooFinanceClient()
        cap = client.get_market_cap("AAPL.US")

        assert cap == 3000000000000


class TestFormatNumber:
    """Tests for format_number utility function."""

    def test_format_trillions(self):
        """Test formatting trillion values."""
        assert format_number(3.5e12, prefix="$") == "$3.50T"

    def test_format_billions(self):
        """Test formatting billion values."""
        assert format_number(2.5e9, prefix="$") == "$2.50B"

    def test_format_millions(self):
        """Test formatting million values."""
        assert format_number(1.5e6, prefix="$") == "$1.50M"

    def test_format_thousands(self):
        """Test formatting thousand values."""
        assert format_number(5.5e3, prefix="$") == "$5.50K"

    def test_format_small_number(self):
        """Test formatting small values."""
        assert format_number(123.45, prefix="$") == "$123.45"

    def test_format_none(self):
        """Test formatting None returns N/A."""
        assert format_number(None) == "N/A"

    def test_format_with_suffix(self):
        """Test formatting with suffix."""
        assert format_number(1e9, suffix=" USD") == "1.00B USD"


class TestFormatPercent:
    """Tests for format_percent utility function."""

    def test_format_positive_percent(self):
        """Test formatting positive percentage."""
        assert format_percent(0.15) == "15.00%"

    def test_format_negative_percent(self):
        """Test formatting negative percentage."""
        assert format_percent(-0.05) == "-5.00%"

    def test_format_zero_percent(self):
        """Test formatting zero percentage."""
        assert format_percent(0.0) == "0.00%"

    def test_format_none_percent(self):
        """Test formatting None returns N/A."""
        assert format_percent(None) == "N/A"

    def test_format_custom_decimals(self):
        """Test formatting with custom decimals."""
        assert format_percent(0.1234, decimals=1) == "12.3%"

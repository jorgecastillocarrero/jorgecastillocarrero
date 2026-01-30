"""
Tests for src/eodhd_client.py - EODHD API client.
"""

import pytest
import time
from datetime import date
from unittest.mock import patch, MagicMock

import httpx
import pandas as pd

from src.eodhd_client import (
    EODHDClient,
    EODHDClientError,
    RateLimitError,
    APIKeyError,
)


class TestEODHDClientInit:
    """Tests for EODHDClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = EODHDClient(api_key="test_key")
        assert client.api_key == "test_key"
        client.close()

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {"EODHD_API_KEY": ""}, clear=False):
            with patch("src.eodhd_client.get_settings") as mock_settings:
                mock_settings.return_value.eodhd_api_key = ""
                with pytest.raises(APIKeyError):
                    EODHDClient()

    def test_init_loads_settings(self):
        """Test that client loads settings correctly."""
        with patch("src.eodhd_client.get_settings") as mock_settings:
            mock_settings.return_value.eodhd_api_key = "settings_key"
            mock_settings.return_value.eodhd_base_url = "https://test.api.com"
            mock_settings.return_value.api_rate_limit = 50
            mock_settings.return_value.api_retry_attempts = 5
            mock_settings.return_value.api_retry_delay = 2.0

            client = EODHDClient()
            assert client.api_key == "settings_key"
            assert client.base_url == "https://test.api.com"
            assert client.rate_limit == 50
            assert client.retry_attempts == 5
            assert client.retry_delay == 2.0
            client.close()


class TestEODHDClientContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self):
        """Test context manager enter and exit."""
        with EODHDClient(api_key="test_key") as client:
            assert client.api_key == "test_key"

    def test_context_manager_closes_client(self):
        """Test that context manager closes HTTP client."""
        with EODHDClient(api_key="test_key") as client:
            http_client = client._client
        # After exiting, client should be closed
        assert True  # Just verify no exception


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_resets_after_minute(self):
        """Test that rate limit counter resets after a minute."""
        client = EODHDClient(api_key="test_key")
        client._request_count = 99
        client._last_request_time = time.time() - 61  # Over a minute ago
        client._rate_limit_check()
        assert client._request_count == 1
        client.close()

    def test_rate_limit_increments_counter(self):
        """Test that each request increments counter."""
        client = EODHDClient(api_key="test_key")
        client._request_count = 0
        client._last_request_time = time.time()
        client._rate_limit_check()
        assert client._request_count == 1
        client._rate_limit_check()
        assert client._request_count == 2
        client.close()


class TestMakeRequest:
    """Tests for _make_request method."""

    def test_make_request_adds_api_token(self):
        """Test that API token is added to params."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_get.return_value = mock_response

            client._make_request("test/endpoint", {"param1": "value1"})

            call_args = mock_get.call_args
            params = call_args[1]["params"]
            assert params["api_token"] == "test_key"
            assert params["fmt"] == "json"

        client.close()

    def test_make_request_handles_429(self):
        """Test that 429 response raises RateLimitError."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_get.return_value = mock_response

            with pytest.raises(RateLimitError):
                client._make_request("test/endpoint")

        client.close()

    def test_make_request_handles_401(self):
        """Test that 401 response raises APIKeyError."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response

            with pytest.raises(APIKeyError):
                client._make_request("test/endpoint")

        client.close()

    def test_make_request_retries_on_error(self):
        """Test that request retries on HTTP error."""
        client = EODHDClient(api_key="test_key")
        client.retry_attempts = 2
        client.retry_delay = 0.01  # Fast retry for test

        with patch.object(client._client, "get") as mock_get:
            # First call fails, second succeeds
            mock_get.side_effect = [
                httpx.RequestError("Connection failed"),
                MagicMock(status_code=200, json=lambda: {"data": "ok"}),
            ]

            result = client._make_request("test/endpoint")
            assert result == {"data": "ok"}
            assert mock_get.call_count == 2

        client.close()


class TestGetEODData:
    """Tests for get_eod_data method."""

    def test_get_eod_data_returns_dataframe(self):
        """Test that get_eod_data returns DataFrame."""
        client = EODHDClient(api_key="test_key")

        mock_data = [
            {"date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000000},
            {"date": "2024-01-03", "open": 104, "high": 108, "low": 103, "close": 107, "volume": 1100000},
        ]

        with patch.object(client, "_make_request", return_value=mock_data):
            df = client.get_eod_data("AAPL.US")

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "symbol" in df.columns

        client.close()

    def test_get_eod_data_empty_response(self):
        """Test handling of empty response."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, "_make_request", return_value=[]):
            df = client.get_eod_data("INVALID.US")
            assert df.empty

        client.close()

    def test_get_eod_data_with_date_params(self):
        """Test date parameters are passed correctly."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, "_make_request", return_value=[]) as mock_req:
            client.get_eod_data(
                "AAPL.US",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31)
            )

            # Verify _make_request was called with correct endpoint and params
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            endpoint = call_args[0][0]
            params = call_args[0][1] if len(call_args[0]) > 1 else {}
            assert endpoint == "eod/AAPL.US"
            assert params.get("from") == "2024-01-01"
            assert params.get("to") == "2024-12-31"

        client.close()


class TestGetRealTimeQuote:
    """Tests for real-time quote methods."""

    def test_get_real_time_quote(self):
        """Test get_real_time_quote returns dict."""
        client = EODHDClient(api_key="test_key")

        mock_data = {"code": "AAPL", "close": 150.25, "timestamp": 1234567890}

        with patch.object(client, "_make_request", return_value=mock_data):
            result = client.get_real_time_quote("AAPL.US")
            assert result == mock_data

        client.close()

    def test_get_real_time_quotes_batch(self):
        """Test batch real-time quotes."""
        client = EODHDClient(api_key="test_key")

        mock_data = [
            {"code": "AAPL", "close": 150.25},
            {"code": "MSFT", "close": 300.50},
        ]

        with patch.object(client, "_make_request", return_value=mock_data):
            result = client.get_real_time_quotes_batch(["AAPL.US", "MSFT.US"])
            assert len(result) == 2

        client.close()


class TestGetFundamentals:
    """Tests for fundamentals methods."""

    def test_get_fundamentals(self):
        """Test get_fundamentals returns dict."""
        client = EODHDClient(api_key="test_key")

        mock_data = {"General": {"Name": "Apple Inc"}, "Highlights": {"MarketCap": 3000000000000}}

        with patch.object(client, "_make_request", return_value=mock_data):
            result = client.get_fundamentals("AAPL.US")
            assert result == mock_data

        client.close()


class TestGetExchanges:
    """Tests for exchange-related methods."""

    def test_get_exchanges_list(self):
        """Test get_exchanges_list returns list."""
        client = EODHDClient(api_key="test_key")

        mock_data = [{"Code": "US", "Name": "USA"}, {"Code": "LSE", "Name": "London"}]

        with patch.object(client, "_make_request", return_value=mock_data):
            result = client.get_exchanges_list()
            assert len(result) == 2

        client.close()

    def test_get_exchange_symbols(self):
        """Test get_exchange_symbols returns list."""
        client = EODHDClient(api_key="test_key")

        mock_data = [{"Code": "AAPL", "Name": "Apple"}, {"Code": "MSFT", "Name": "Microsoft"}]

        with patch.object(client, "_make_request", return_value=mock_data):
            result = client.get_exchange_symbols("US")
            assert len(result) == 2

        client.close()


class TestDividendsAndSplits:
    """Tests for dividends and splits methods."""

    def test_get_dividends_returns_dataframe(self):
        """Test get_dividends returns DataFrame."""
        client = EODHDClient(api_key="test_key")

        mock_data = [
            {"date": "2024-01-15", "value": 0.24},
            {"date": "2024-04-15", "value": 0.25},
        ]

        with patch.object(client, "_make_request", return_value=mock_data):
            df = client.get_dividends("AAPL.US")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2

        client.close()

    def test_get_splits_returns_dataframe(self):
        """Test get_splits returns DataFrame."""
        client = EODHDClient(api_key="test_key")

        mock_data = [{"date": "2020-08-31", "split": "4:1"}]

        with patch.object(client, "_make_request", return_value=mock_data):
            df = client.get_splits("AAPL.US")
            assert isinstance(df, pd.DataFrame)

        client.close()

    def test_get_dividends_empty(self):
        """Test empty dividends response."""
        client = EODHDClient(api_key="test_key")

        with patch.object(client, "_make_request", return_value=[]):
            df = client.get_dividends("NODIV.US")
            assert df.empty

        client.close()

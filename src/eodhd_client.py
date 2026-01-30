"""
EODHD API Client for financial data retrieval.
Supports EOD data, real-time quotes, fundamentals, and bulk downloads.
"""

import logging
import time
from datetime import date, datetime
from typing import Any

import httpx
import pandas as pd

from .config import get_settings

logger = logging.getLogger(__name__)


class EODHDClientError(Exception):
    """Base exception for EODHD client errors."""

    pass


class RateLimitError(EODHDClientError):
    """Raised when API rate limit is exceeded."""

    pass


class APIKeyError(EODHDClientError):
    """Raised when API key is invalid or missing."""

    pass


class EODHDClient:
    """Client for interacting with EODHD API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the EODHD client.

        Args:
            api_key: EODHD API key. If not provided, uses settings.
        """
        settings = get_settings()
        self.api_key = api_key or settings.eodhd_api_key
        self.base_url = settings.eodhd_base_url
        self.rate_limit = settings.api_rate_limit
        self.retry_attempts = settings.api_retry_attempts
        self.retry_delay = settings.api_retry_delay

        if not self.api_key:
            raise APIKeyError("EODHD API key is required")

        self._client = httpx.Client(timeout=30.0)
        self._last_request_time = 0.0
        self._request_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def _rate_limit_check(self):
        """Implement rate limiting."""
        current_time = time.time()
        if current_time - self._last_request_time > 60:
            self._request_count = 0
            self._last_request_time = current_time

        if self._request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self._last_request_time)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._request_count = 0
                self._last_request_time = time.time()

        self._request_count += 1

    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict | list:
        """
        Make an API request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            EODHDClientError: On API errors
        """
        self._rate_limit_check()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["api_token"] = self.api_key
        params["fmt"] = "json"

        for attempt in range(self.retry_attempts):
            try:
                response = self._client.get(url, params=params)

                if response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")

                if response.status_code == 401:
                    raise APIKeyError("Invalid API key")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise EODHDClientError(f"API request failed: {e}")

            except httpx.RequestError as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise EODHDClientError(f"Request failed: {e}")

    # =========================================================================
    # End-of-Day Data
    # =========================================================================

    def get_eod_data(
        self,
        symbol: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        period: str = "d",
    ) -> pd.DataFrame:
        """
        Get historical end-of-day data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')
            start_date: Start date for historical data
            end_date: End date for historical data
            period: Data period - 'd' (daily), 'w' (weekly), 'm' (monthly)

        Returns:
            DataFrame with OHLCV data
        """
        params = {"period": period}

        if start_date:
            params["from"] = (
                start_date.isoformat()
                if isinstance(start_date, date)
                else start_date
            )
        if end_date:
            params["to"] = (
                end_date.isoformat() if isinstance(end_date, date) else end_date
            )

        data = self._make_request(f"eod/{symbol}", params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            df = df.set_index("date").sort_index()

        return df

    # =========================================================================
    # Real-Time Data
    # =========================================================================

    def get_real_time_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')

        Returns:
            Dictionary with real-time quote data
        """
        data = self._make_request(f"real-time/{symbol}")
        return data

    def get_real_time_quotes_batch(self, symbols: list[str]) -> list[dict]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            List of quote dictionaries
        """
        symbols_str = ",".join(symbols)
        params = {"s": symbols_str}
        data = self._make_request("real-time/", params)
        return data if isinstance(data, list) else [data]

    # =========================================================================
    # Fundamental Data
    # =========================================================================

    def get_fundamentals(self, symbol: str) -> dict:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL.US')

        Returns:
            Dictionary with fundamental data
        """
        data = self._make_request(f"fundamentals/{symbol}")
        return data

    def get_fundamentals_bulk(
        self, exchange: str, symbols: list[str] | None = None
    ) -> list[dict]:
        """
        Get bulk fundamental data for an exchange.

        Args:
            exchange: Exchange code (e.g., 'US')
            symbols: Optional list of specific symbols

        Returns:
            List of fundamental data dictionaries
        """
        params = {}
        if symbols:
            params["symbols"] = ",".join(symbols)

        data = self._make_request(f"bulk-fundamentals/{exchange}", params)
        return data if isinstance(data, list) else [data]

    # =========================================================================
    # Exchange Data
    # =========================================================================

    def get_exchanges_list(self) -> list[dict]:
        """
        Get list of available exchanges.

        Returns:
            List of exchange dictionaries
        """
        data = self._make_request("exchanges-list")
        return data

    def get_exchange_symbols(
        self, exchange: str, symbol_type: str | None = None
    ) -> list[dict]:
        """
        Get list of symbols for an exchange.

        Args:
            exchange: Exchange code (e.g., 'US')
            symbol_type: Optional filter by type ('stock', 'etf', 'fund', etc.)

        Returns:
            List of symbol dictionaries
        """
        params = {}
        if symbol_type:
            params["type"] = symbol_type

        data = self._make_request(f"exchange-symbol-list/{exchange}", params)
        return data

    # =========================================================================
    # Bulk EOD Data
    # =========================================================================

    def get_eod_bulk(
        self, exchange: str, date_str: str | None = None
    ) -> pd.DataFrame:
        """
        Get bulk EOD data for an entire exchange.

        Args:
            exchange: Exchange code (e.g., 'US')
            date_str: Optional specific date (YYYY-MM-DD)

        Returns:
            DataFrame with bulk EOD data
        """
        params = {}
        if date_str:
            params["date"] = date_str

        data = self._make_request(f"eod-bulk-last-day/{exchange}", params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df

    # =========================================================================
    # Dividends and Splits
    # =========================================================================

    def get_dividends(
        self,
        symbol: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """
        Get dividend data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with dividend data
        """
        params = {}
        if start_date:
            params["from"] = (
                start_date.isoformat()
                if isinstance(start_date, date)
                else start_date
            )
        if end_date:
            params["to"] = (
                end_date.isoformat() if isinstance(end_date, date) else end_date
            )

        data = self._make_request(f"div/{symbol}", params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol

        return df

    def get_splits(
        self,
        symbol: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> pd.DataFrame:
        """
        Get stock split data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with split data
        """
        params = {}
        if start_date:
            params["from"] = (
                start_date.isoformat()
                if isinstance(start_date, date)
                else start_date
            )
        if end_date:
            params["to"] = (
                end_date.isoformat() if isinstance(end_date, date) else end_date
            )

        data = self._make_request(f"splits/{symbol}", params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol

        return df


# CLI test functionality
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    settings = get_settings()
    if not settings.is_eodhd_configured:
        print("Error: EODHD_API_KEY not configured. Set it in .env file.")
        sys.exit(1)

    with EODHDClient() as client:
        print("\n=== Testing EODHD Client ===\n")

        # Test exchanges list
        print("Fetching exchanges list...")
        exchanges = client.get_exchanges_list()
        print(f"Found {len(exchanges)} exchanges")

        # Test EOD data
        print("\nFetching AAPL.US EOD data...")
        df = client.get_eod_data("AAPL.US", start_date="2024-01-01")
        print(f"Retrieved {len(df)} rows of EOD data")
        if not df.empty:
            print(df.tail())

        print("\n=== Tests completed ===")

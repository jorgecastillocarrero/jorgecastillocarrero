"""
Yahoo Finance client for financial data.
Provides historical prices, fundamentals, and market data - all free.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Client for fetching all financial data from Yahoo Finance."""

    @staticmethod
    def _convert_symbol(symbol: str) -> str:
        """Convert EODHD symbol format to Yahoo format."""
        # AAPL.US -> AAPL, TSLA.US -> TSLA
        if ".US" in symbol:
            return symbol.replace(".US", "")
        if "." in symbol:
            parts = symbol.split(".")
            # Handle international symbols
            exchange_map = {
                "LSE": ".L",
                "XETRA": ".DE",
                "PA": ".PA",
                "MC": ".MC",
                # Yahoo native formats - keep as-is
                "TO": ".TO",    # Toronto
                "SW": ".SW",    # Swiss
                "MI": ".MI",    # Milan
                "L": ".L",      # London
                "DE": ".DE",    # Germany
                "F": ".F",      # Frankfurt
                "AS": ".AS",    # Amsterdam
                "BR": ".BR",    # Brussels
                "VI": ".VI",    # Vienna
                "ST": ".ST",    # Stockholm
                "OL": ".OL",    # Oslo
                "CO": ".CO",    # Copenhagen
                "HE": ".HE",    # Helsinki
            }
            if parts[1] in exchange_map:
                return parts[0] + exchange_map[parts[1]]
            return parts[0]
        return symbol

    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a yfinance Ticker object."""
        yahoo_symbol = self._convert_symbol(symbol)
        return yf.Ticker(yahoo_symbol)

    # =========================================================================
    # Historical Price Data
    # =========================================================================

    def get_historical_data(
        self,
        symbol: str,
        period: str = "max",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start: Start date (overrides period)
            end: End date
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = self.get_ticker(symbol)

        try:
            if start:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Rename columns to lowercase
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Add symbol column
        df["symbol"] = symbol

        # Rename 'adj close' or similar to 'adjusted_close'
        if "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "adjusted_close"})
        elif "close" in df.columns and "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        return df

    def get_multiple_historical(
        self,
        symbols: list[str],
        period: str = "max",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        yahoo_symbols = [self._convert_symbol(s) for s in symbols]

        try:
            # Download all at once for efficiency
            data = yf.download(
                yahoo_symbols,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
            )

            for i, symbol in enumerate(symbols):
                yahoo_sym = yahoo_symbols[i]
                if len(symbols) == 1:
                    df = data.copy()
                else:
                    if yahoo_sym in data.columns.get_level_values(0):
                        df = data[yahoo_sym].copy()
                    else:
                        df = pd.DataFrame()

                if not df.empty:
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df["symbol"] = symbol
                    if "adj_close" in df.columns:
                        df = df.rename(columns={"adj_close": "adjusted_close"})
                    elif "close" in df.columns:
                        df["adjusted_close"] = df["close"]
                    df = df.dropna(subset=["close"])

                results[symbol] = df

        except Exception as e:
            logger.error(f"Error in batch download: {e}")
            # Fallback to individual downloads
            for symbol in symbols:
                results[symbol] = self.get_historical_data(symbol, period=period, interval=interval)

        return results

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """Get dividend history."""
        ticker = self.get_ticker(symbol)
        return ticker.dividends.to_frame(name="dividend")

    def get_splits(self, symbol: str) -> pd.DataFrame:
        """Get stock split history."""
        ticker = self.get_ticker(symbol)
        return ticker.splits.to_frame(name="split")

    def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol (EODHD or Yahoo format)

        Returns:
            Dictionary with fundamental data
        """
        yahoo_symbol = self._convert_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            info = ticker.info
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {"error": str(e)}

        return {
            "symbol": symbol,
            "yahoo_symbol": yahoo_symbol,
            "general": {
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "website": info.get("website"),
                "employees": info.get("fullTimeEmployees"),
                "description": info.get("longBusinessSummary"),
            },
            "market_data": {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose"),
                "open": info.get("open") or info.get("regularMarketOpen"),
                "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "avg_volume": info.get("averageVolume"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_avg": info.get("fiftyDayAverage"),
                "200_day_avg": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
            },
            "valuation": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
            },
            "financials": {
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "free_cash_flow": info.get("freeCashflow"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
            },
            "per_share": {
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "book_value": info.get("bookValue"),
                "revenue_per_share": info.get("revenuePerShare"),
            },
            "margins": {
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
            },
            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            },
            "dividend": {
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": info.get("exDividendDate"),
            },
            "returns": {
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
            },
            "analyst": {
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_median": info.get("targetMedianPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            },
        }

    def get_market_cap(self, symbol: str) -> float | None:
        """Get market cap for a symbol."""
        data = self.get_fundamentals(symbol)
        return data.get("market_data", {}).get("market_cap")

    def get_quick_stats(self, symbol: str) -> dict:
        """Get quick fundamental stats."""
        yahoo_symbol = self._convert_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            info = ticker.info
        except Exception as e:
            return {"error": str(e)}

        market_cap = info.get("marketCap", 0)

        # Format market cap
        if market_cap >= 1e12:
            market_cap_str = f"${market_cap / 1e12:.2f}T"
        elif market_cap >= 1e9:
            market_cap_str = f"${market_cap / 1e9:.2f}B"
        elif market_cap >= 1e6:
            market_cap_str = f"${market_cap / 1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:,.0f}"

        return {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName"),
            "market_cap": market_cap,
            "market_cap_formatted": market_cap_str,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "target_price": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
        }


def format_number(value, prefix="", suffix="", decimals=2):
    """Format large numbers with B/M/K suffixes."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"{prefix}{value / 1e12:.{decimals}f}T{suffix}"
    elif abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.{decimals}f}B{suffix}"
    elif abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.{decimals}f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{prefix}{value / 1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{value:.{decimals}f}{suffix}"


def format_percent(value, decimals=2):
    """Format as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


# CLI test
if __name__ == "__main__":
    client = YahooFinanceClient()

    print("\n=== Testing Yahoo Finance Client ===\n")

    stats = client.get_quick_stats("NVDA.US")
    print(f"Symbol: {stats['symbol']}")
    print(f"Name: {stats['name']}")
    print(f"Market Cap: {stats['market_cap_formatted']}")
    print(f"P/E Ratio: {stats['pe_ratio']}")
    print(f"EPS: ${stats['eps']}")

    print("\n=== Test completed ===")

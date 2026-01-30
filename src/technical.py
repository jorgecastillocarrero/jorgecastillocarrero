"""
Technical analysis calculations module.
Calculates RSI, moving averages, returns, Sharpe ratio, etc.
"""

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .database import get_db_manager, Symbol, PriceHistory

logger = logging.getLogger(__name__)

# Risk-free rate for Sharpe calculation (annualized)
RISK_FREE_RATE = 0.05  # 5% annual


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series with RSI values
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use exponential moving average for smoother RSI
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate simple moving average.

    Args:
        prices: Series of closing prices
        period: MA period

    Returns:
        Series with MA values
    """
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_distance_from_ma(prices: pd.Series, ma: pd.Series) -> pd.Series:
    """
    Calculate percentage distance from moving average.

    Args:
        prices: Series of closing prices
        ma: Series of moving average values

    Returns:
        Series with distance values (as decimal, e.g., 0.05 = 5% above MA)
    """
    return (prices - ma) / ma


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate percentage returns.

    Args:
        prices: Series of closing prices
        periods: Number of periods for return calculation

    Returns:
        Series with return values (as decimal)
    """
    return prices.pct_change(periods=periods)


def calculate_ytd_return(prices: pd.Series) -> pd.Series:
    """
    Calculate year-to-date return for each date.

    Args:
        prices: Series of closing prices with DatetimeIndex

    Returns:
        Series with YTD return values
    """
    ytd_returns = pd.Series(index=prices.index, dtype=float)

    for date in prices.index:
        year_start = datetime(date.year, 1, 1)
        # Find first trading day of the year
        year_prices = prices[prices.index >= pd.Timestamp(year_start)]
        if len(year_prices) > 0:
            first_price = year_prices.iloc[0]
            current_price = prices[date]
            ytd_returns[date] = (current_price - first_price) / first_price

    return ytd_returns


def calculate_volatility(returns: pd.Series, period: int = 21) -> pd.Series:
    """
    Calculate rolling annualized volatility.

    Args:
        returns: Series of daily returns
        period: Rolling window period

    Returns:
        Series with annualized volatility values
    """
    # Annualize: multiply by sqrt(252 trading days)
    return returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)


def calculate_sharpe_ratio(
    returns: pd.Series, period: int = 21, risk_free_rate: float = RISK_FREE_RATE
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Series of daily returns
        period: Rolling window period
        risk_free_rate: Annual risk-free rate

    Returns:
        Series with Sharpe ratio values
    """
    # Daily risk-free rate
    daily_rf = risk_free_rate / 252

    excess_returns = returns - daily_rf
    avg_excess = excess_returns.rolling(window=period, min_periods=period).mean()
    volatility = returns.rolling(window=period, min_periods=period).std()

    # Annualize Sharpe
    sharpe = (avg_excess / volatility) * np.sqrt(252)

    return sharpe


def calculate_all_metrics(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical metrics for a price DataFrame.

    Args:
        prices_df: DataFrame with OHLCV data (must have 'close' or 'adjusted_close' column)

    Returns:
        DataFrame with all calculated metrics
    """
    if prices_df.empty:
        return pd.DataFrame()

    # Use adjusted close if available, otherwise close
    if "adjusted_close" in prices_df.columns and prices_df["adjusted_close"].notna().any():
        close = prices_df["adjusted_close"].fillna(prices_df["close"])
    else:
        close = prices_df["close"]

    metrics = pd.DataFrame(index=prices_df.index)

    # Price
    metrics["close_price"] = close

    # RSI
    metrics["rsi_14"] = calculate_rsi(close, 14)

    # Moving Averages
    metrics["ma_50"] = calculate_moving_average(close, 50)
    metrics["ma_200"] = calculate_moving_average(close, 200)

    # Distance from MAs
    metrics["m50"] = calculate_distance_from_ma(close, metrics["ma_50"])
    metrics["m200"] = calculate_distance_from_ma(close, metrics["ma_200"])

    # Returns
    metrics["daily_return"] = calculate_returns(close, 1)
    metrics["weekly_return"] = calculate_returns(close, 5)
    metrics["monthly_return"] = calculate_returns(close, 21)

    # YTD Return
    metrics["ytd_return"] = calculate_ytd_return(close)

    # Volatility and Sharpe
    metrics["volatility_21d"] = calculate_volatility(metrics["daily_return"], 21)
    metrics["sharpe_21d"] = calculate_sharpe_ratio(metrics["daily_return"], 21)

    return metrics


class MetricsCalculator:
    """Calculates and stores technical metrics for symbols."""

    def __init__(self):
        self.db = get_db_manager()

    def calculate_for_symbol(
        self,
        symbol_code: str,
        start_date: datetime = None,
        end_date: datetime = None,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate metrics for a single symbol.

        Args:
            symbol_code: Stock symbol code
            start_date: Start date for calculation
            end_date: End date for calculation
            save_to_db: Whether to save results to database

        Returns:
            DataFrame with calculated metrics
        """
        with self.db.get_session() as session:
            # Get symbol
            symbol = session.query(Symbol).filter(Symbol.code == symbol_code).first()
            if not symbol:
                logger.warning(f"{symbol_code}: Symbol not found")
                return pd.DataFrame()

            # Get price history (need at least 200 days for MA200)
            # Get extra historical data for accurate calculations
            actual_start = start_date
            if start_date:
                actual_start = start_date - timedelta(days=250)

            prices_df = self.db.get_price_history(
                session, symbol.id, actual_start, end_date
            )

            if prices_df.empty or len(prices_df) < 200:
                logger.warning(f"{symbol_code}: Insufficient price data ({len(prices_df)} records)")
                return pd.DataFrame()

            # Calculate metrics
            metrics_df = calculate_all_metrics(prices_df)

            # Filter to requested date range
            if start_date:
                metrics_df = metrics_df[metrics_df.index >= pd.Timestamp(start_date)]
            if end_date:
                metrics_df = metrics_df[metrics_df.index <= pd.Timestamp(end_date)]

            # Save to database if requested
            if save_to_db and not metrics_df.empty:
                count = self.db.bulk_upsert_daily_metrics(session, symbol.id, metrics_df)
                logger.info(f"{symbol_code}: Saved {count} daily metrics")

            return metrics_df

    def calculate_for_all_symbols(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        symbols: list[str] = None,
    ) -> dict:
        """
        Calculate metrics for all (or specified) symbols.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            symbols: List of symbol codes (all if None)

        Returns:
            Results summary dict
        """
        with self.db.get_session() as session:
            if symbols:
                symbol_list = (
                    session.query(Symbol)
                    .filter(Symbol.code.in_(symbols))
                    .all()
                )
            else:
                symbol_list = session.query(Symbol).all()

        results = {
            "total": len(symbol_list),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }

        logger.info(f"Calculating metrics for {results['total']} symbols")

        for i, symbol in enumerate(symbol_list, 1):
            try:
                metrics_df = self.calculate_for_symbol(
                    symbol.code, start_date, end_date, save_to_db=True
                )

                if metrics_df.empty:
                    results["skipped"] += 1
                else:
                    results["success"] += 1

                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{results['total']} | "
                        f"Success: {results['success']} | "
                        f"Skipped: {results['skipped']}"
                    )

            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"symbol": symbol.code, "error": str(e)[:100]})
                logger.warning(f"{symbol.code}: Error - {str(e)[:50]}")

        logger.info(
            f"Metrics calculation completed: "
            f"{results['success']} success, {results['skipped']} skipped, "
            f"{results['failed']} failed"
        )

        return results

    def get_latest_metrics(
        self, symbol_codes: list[str] = None
    ) -> pd.DataFrame:
        """
        Get the most recent metrics for symbols.

        Args:
            symbol_codes: List of symbol codes (all if None)

        Returns:
            DataFrame with latest metrics for each symbol
        """
        with self.db.get_session() as session:
            if symbol_codes:
                symbols = (
                    session.query(Symbol)
                    .filter(Symbol.code.in_(symbol_codes))
                    .all()
                )
                symbol_ids = [s.id for s in symbols]
            else:
                symbol_ids = None

            return self.db.get_latest_metrics_for_symbols(session, symbol_ids)


# CLI functionality
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    calculator = MetricsCalculator()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Calculate for all symbols
            print("\n=== Calculating Metrics for All Symbols ===\n")
            results = calculator.calculate_for_all_symbols()
            print(f"\nResults:")
            print(f"  Success: {results['success']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Failed: {results['failed']}")

        elif sys.argv[1] == "--recent":
            # Calculate only for recent dates (last 30 days)
            print("\n=== Calculating Recent Metrics (30 days) ===\n")
            start = datetime.now() - timedelta(days=30)
            results = calculator.calculate_for_all_symbols(start_date=start)
            print(f"\nResults:")
            print(f"  Success: {results['success']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Failed: {results['failed']}")

        else:
            # Calculate for specific symbols
            symbols = sys.argv[1:]
            print(f"\n=== Calculating Metrics for {symbols} ===\n")
            for symbol in symbols:
                metrics_df = calculator.calculate_for_symbol(symbol)
                if not metrics_df.empty:
                    print(f"\n{symbol} - Latest metrics:")
                    print(metrics_df.tail(1).T)

    else:
        # Default: show usage
        print("Usage:")
        print("  python -m src.technical --all      # Calculate for all symbols")
        print("  python -m src.technical --recent   # Calculate last 30 days only")
        print("  python -m src.technical AAPL MSFT  # Calculate for specific symbols")

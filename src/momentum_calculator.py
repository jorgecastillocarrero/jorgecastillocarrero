"""
Momentum Features Calculator
Calculates returns, risk metrics, and forward returns (ML targets).

Usage:
    python -m src.momentum_calculator --symbol AAPL
    python -m src.momentum_calculator --all --limit 100
    python -m src.momentum_calculator --test
"""

import logging
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
RISK_FREE_RATE = 0.05  # 5% annual


class MomentumCalculator:
    """Calculate momentum and risk features."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url
        self._market_returns = None  # Cache for SPY returns

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_market_returns(self) -> pd.Series:
        """Get SPY returns as market benchmark (cached)."""
        if self._market_returns is not None:
            return self._market_returns

        conn = self.get_connection()
        try:
            query = """
                SELECT date, close
                FROM fmp_price_history
                WHERE symbol = 'SPY'
                ORDER BY date
            """
            df = pd.read_sql(query, conn, params=())
            if df.empty:
                logger.warning("SPY data not found, using ^GSPC")
                query = query.replace("'SPY'", "'^GSPC'")
                df = pd.read_sql(query, conn, params=())

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self._market_returns = df['close'].pct_change()
            else:
                self._market_returns = pd.Series(dtype=float)

            return self._market_returns
        finally:
            conn.close()

    def get_price_data(self, symbol: str, min_days: int = 300) -> pd.DataFrame:
        """Fetch close prices for a symbol."""
        conn = self.get_connection()
        try:
            query = """
                SELECT date, close
                FROM fmp_price_history
                WHERE symbol = %s
                ORDER BY date
            """
            df = pd.read_sql(query, conn, params=(symbol,))
            if len(df) < min_days:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_all_symbols(self, limit: Optional[int] = None) -> list:
        """Get symbols with sufficient data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT symbol FROM fmp_price_history
                GROUP BY symbol
                HAVING COUNT(*) >= 300
                ORDER BY symbol
            """
            if limit:
                query = query.replace("ORDER BY symbol", f"ORDER BY symbol LIMIT {limit}")
            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def calculate_returns(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate returns over period."""
        return close.pct_change(periods=period)

    def calculate_forward_returns(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate FORWARD returns (shift negative = future)."""
        return close.pct_change(periods=period).shift(-period)

    def calculate_volatility(self, returns: pd.Series, period: int) -> pd.Series:
        """Calculate annualized rolling volatility."""
        return returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)

    def calculate_sharpe(self, returns: pd.Series, period: int, rf_rate: float = RISK_FREE_RATE) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        daily_rf = rf_rate / 252
        excess = returns - daily_rf
        avg_excess = excess.rolling(window=period, min_periods=period).mean()
        vol = returns.rolling(window=period, min_periods=period).std()
        return (avg_excess / vol) * np.sqrt(252)

    def calculate_sortino(self, returns: pd.Series, period: int, rf_rate: float = RISK_FREE_RATE) -> pd.Series:
        """Calculate rolling Sortino ratio (downside deviation only)."""
        daily_rf = rf_rate / 252
        excess = returns - daily_rf
        avg_excess = excess.rolling(window=period, min_periods=period).mean()

        # Downside deviation
        negative_returns = returns.where(returns < 0, 0)
        downside_vol = negative_returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)

        return avg_excess * np.sqrt(252) / downside_vol

    def calculate_max_drawdown(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate rolling max drawdown."""
        rolling_max = close.rolling(window=period, min_periods=period).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown

    def calculate_momentum_score(self, ret_20d: pd.Series, ret_60d: pd.Series,
                                   ret_252d: pd.Series, vol_20d: pd.Series) -> pd.Series:
        """
        Calculate composite momentum score.
        Higher = better risk-adjusted momentum.
        """
        # Normalize returns by volatility (risk-adjusted)
        score = (
            (ret_20d / vol_20d.replace(0, np.nan)) * 0.4 +
            (ret_60d / vol_20d.replace(0, np.nan)) * 0.3 +
            (ret_252d / vol_20d.replace(0, np.nan)) * 0.3
        )
        return score

    def calculate_rolling_beta(self, stock_returns: pd.Series, market_returns: pd.Series,
                                window: int) -> pd.Series:
        """
        Calculate rolling beta vs market.
        Beta = Cov(stock, market) / Var(market)
        """
        # Align the series
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if len(aligned) < window:
            return pd.Series(index=stock_returns.index, dtype=float)

        # Rolling covariance and variance
        cov = aligned['stock'].rolling(window=window, min_periods=window).cov(aligned['market'])
        var = aligned['market'].rolling(window=window, min_periods=window).var()

        beta = cov / var

        # Reindex to original stock index
        return beta.reindex(stock_returns.index)

    def classify_beta_zone(self, beta: pd.Series) -> pd.Series:
        """Classify beta into zones."""
        conditions = [
            beta < 0.5,
            (beta >= 0.5) & (beta < 0.8),
            (beta >= 0.8) & (beta < 1.2),
            (beta >= 1.2) & (beta < 1.5),
            beta >= 1.5
        ]
        choices = ['very_low', 'low', 'market', 'high', 'very_high']
        return pd.Series(np.select(conditions, choices, default=None), index=beta.index)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum features."""
        close = df['close']
        returns = close.pct_change()

        f = pd.DataFrame(index=df.index)

        # Historical returns
        f['ret_1d'] = self.calculate_returns(close, 1)
        f['ret_5d'] = self.calculate_returns(close, 5)
        f['ret_20d'] = self.calculate_returns(close, 20)
        f['ret_60d'] = self.calculate_returns(close, 60)
        f['ret_252d'] = self.calculate_returns(close, 252)

        # Forward returns (ML TARGETS)
        f['ret_1d_fwd'] = self.calculate_forward_returns(close, 1)
        f['ret_5d_fwd'] = self.calculate_forward_returns(close, 5)
        f['ret_20d_fwd'] = self.calculate_forward_returns(close, 20)

        # Volatility
        f['vol_20d'] = self.calculate_volatility(returns, 20)
        f['vol_60d'] = self.calculate_volatility(returns, 60)

        # Risk metrics
        f['sharpe_20d'] = self.calculate_sharpe(returns, 20)
        f['sharpe_60d'] = self.calculate_sharpe(returns, 60)
        f['sortino_20d'] = self.calculate_sortino(returns, 20)
        f['sortino_60d'] = self.calculate_sortino(returns, 60)

        # Max drawdown
        f['max_dd_20d'] = self.calculate_max_drawdown(close, 20)
        f['max_dd_60d'] = self.calculate_max_drawdown(close, 60)

        # Composite momentum score
        f['momentum_score'] = self.calculate_momentum_score(
            f['ret_20d'], f['ret_60d'], f['ret_252d'], f['vol_20d']
        )

        # Rolling Betas vs SPY
        market_returns = self.get_market_returns()
        if not market_returns.empty:
            f['beta_20d'] = self.calculate_rolling_beta(returns, market_returns, 20)
            f['beta_60d'] = self.calculate_rolling_beta(returns, market_returns, 60)
            f['beta_120d'] = self.calculate_rolling_beta(returns, market_returns, 120)
            f['beta_252d'] = self.calculate_rolling_beta(returns, market_returns, 252)
            f['beta_zone'] = self.classify_beta_zone(f['beta_60d'])  # Use 60d as default zone
        else:
            logger.warning("No market data available for beta calculation")
            f['beta_20d'] = None
            f['beta_60d'] = None
            f['beta_120d'] = None
            f['beta_252d'] = None
            f['beta_zone'] = None

        return f

    def save_features(self, symbol: str, features: pd.DataFrame) -> int:
        """Save features to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM features_momentum WHERE symbol = %s", (symbol,))

            columns = ['symbol', 'date'] + list(features.columns)
            values = []

            for date, row in features.iterrows():
                record = [symbol, date.date() if hasattr(date, 'date') else date]
                for col in features.columns:
                    val = row[col]
                    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                        record.append(None)
                    elif isinstance(val, (np.floating, float)):
                        record.append(float(val))
                    elif isinstance(val, (np.integer, int)):
                        record.append(int(val))
                    else:
                        record.append(val)
                values.append(tuple(record))

            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO features_momentum ({', '.join(columns)}) VALUES ({placeholders})"
            cur.executemany(query, values)
            conn.commit()
            return len(values)
        finally:
            conn.close()

    def process_symbol(self, symbol: str) -> bool:
        """Process single symbol."""
        try:
            df = self.get_price_data(symbol)
            if df.empty:
                logger.warning(f"{symbol}: Insufficient data")
                return False

            features = self.calculate_features(df)
            features = features.dropna(thresh=len(features.columns) * 0.3)

            if features.empty:
                logger.warning(f"{symbol}: No valid features")
                return False

            count = self.save_features(symbol, features)
            logger.info(f"{symbol}: Saved {count} records")
            return True
        except Exception as e:
            logger.error(f"{symbol}: {e}")
            return False

    def process_all(self, limit: Optional[int] = None, batch_log: int = 50):
        """Process all symbols."""
        symbols = self.get_all_symbols(limit)
        total, success, failed = len(symbols), 0, 0

        logger.info(f"Processing {total} symbols...")

        for i, symbol in enumerate(symbols, 1):
            if self.process_symbol(symbol):
                success += 1
            else:
                failed += 1

            if i % batch_log == 0:
                logger.info(f"Progress: {i}/{total} | OK: {success} | Failed: {failed}")

        logger.info(f"Done: {success} OK, {failed} failed")
        return {'total': total, 'success': success, 'failed': failed}


def main():
    parser = argparse.ArgumentParser(description='Calculate momentum features')
    parser.add_argument('--symbol', type=str, help='Single symbol')
    parser.add_argument('--all', action='store_true', help='All symbols')
    parser.add_argument('--limit', type=int, help='Limit symbols')
    parser.add_argument('--test', action='store_true', help='Test with AAPL')

    args = parser.parse_args()
    calc = MomentumCalculator()

    if args.test:
        calc.process_symbol('AAPL')
    elif args.symbol:
        calc.process_symbol(args.symbol)
    elif args.all:
        calc.process_all(limit=args.limit)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

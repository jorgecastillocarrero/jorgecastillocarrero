"""
Earnings Features Calculator
Calculates earnings surprise, days to earnings, and beat streaks.

Usage:
    python -m src.earnings_calculator --symbol AAPL
    python -m src.earnings_calculator --all --limit 100
    python -m src.earnings_calculator --test
"""

import logging
import argparse
from typing import Optional
from datetime import timedelta
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class EarningsCalculator:
    """Calculate earnings-related features."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_trading_dates(self, symbol: str) -> list:
        """Get all trading dates for symbol."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute('''
                SELECT DISTINCT date FROM fmp_price_history
                WHERE symbol = %s ORDER BY date
            ''', (symbol,))
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_earnings_data(self, symbol: str) -> pd.DataFrame:
        """Get earnings reports for symbol."""
        conn = self.get_connection()
        try:
            query = '''
                SELECT date, eps_actual, eps_estimated, revenue_actual, revenue_estimated
                FROM fmp_earnings
                WHERE symbol = %s
                ORDER BY date
            '''
            df = pd.read_sql(query, conn, params=(symbol,))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            return df
        finally:
            conn.close()

    def get_all_symbols(self, limit: Optional[int] = None) -> list:
        """Get symbols with earnings data."""
        conn = self.get_connection()
        try:
            query = '''
                SELECT DISTINCT e.symbol
                FROM fmp_earnings e
                JOIN fmp_price_history p ON e.symbol = p.symbol
                GROUP BY e.symbol
                HAVING COUNT(DISTINCT e.date) >= 4
                ORDER BY e.symbol
            '''
            if limit:
                query = query.replace("ORDER BY e.symbol", f"ORDER BY e.symbol LIMIT {limit}")
            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def calculate_surprise(self, actual, estimated) -> Optional[float]:
        """Calculate earnings/revenue surprise as percentage."""
        if actual is None or estimated is None or estimated == 0:
            return None
        return (actual - estimated) / abs(estimated)

    def calculate_beat_streak(self, earnings_df: pd.DataFrame, trade_date) -> int:
        """Calculate consecutive quarters beating estimates."""
        past = earnings_df[earnings_df['date'] <= trade_date].sort_values('date', ascending=False)
        if past.empty:
            return 0

        streak = 0
        for _, row in past.iterrows():
            if row['eps_actual'] is not None and row['eps_estimated'] is not None:
                if row['eps_actual'] > row['eps_estimated']:
                    streak += 1
                else:
                    break
        return streak

    def calculate_features(self, symbol: str) -> pd.DataFrame:
        """Calculate daily earnings features."""
        trading_dates = self.get_trading_dates(symbol)
        if not trading_dates:
            return pd.DataFrame()

        earnings_df = self.get_earnings_data(symbol)
        if earnings_df.empty:
            return pd.DataFrame()

        features = []

        for trade_date in trading_dates:
            td = pd.Timestamp(trade_date)

            # Past earnings (for surprises)
            past_earnings = earnings_df[earnings_df['date'] <= td].sort_values('date', ascending=False)

            # Future earnings (for days_to_earnings)
            future_earnings = earnings_df[earnings_df['date'] > td].sort_values('date')

            # Days to next earnings
            if not future_earnings.empty:
                next_date = future_earnings.iloc[0]['date']
                days_to = (next_date - td).days
                earnings_date_next = next_date.date()
            else:
                days_to = None
                earnings_date_next = None

            # Last earnings surprise
            earnings_surprise_last = None
            revenue_surprise_last = None
            if not past_earnings.empty:
                last = past_earnings.iloc[0]
                earnings_surprise_last = self.calculate_surprise(last['eps_actual'], last['eps_estimated'])
                revenue_surprise_last = self.calculate_surprise(last['revenue_actual'], last['revenue_estimated'])

            # Average surprise last 4 quarters
            earnings_surprise_avg_4q = None
            revenue_surprise_avg_4q = None
            if len(past_earnings) >= 4:
                last_4 = past_earnings.head(4)
                eps_surprises = [self.calculate_surprise(r['eps_actual'], r['eps_estimated'])
                                 for _, r in last_4.iterrows()]
                rev_surprises = [self.calculate_surprise(r['revenue_actual'], r['revenue_estimated'])
                                 for _, r in last_4.iterrows()]

                eps_surprises = [s for s in eps_surprises if s is not None]
                rev_surprises = [s for s in rev_surprises if s is not None]

                if eps_surprises:
                    earnings_surprise_avg_4q = np.mean(eps_surprises)
                if rev_surprises:
                    revenue_surprise_avg_4q = np.mean(rev_surprises)

            # Beat streak
            beat_streak = self.calculate_beat_streak(earnings_df, td)

            row = {
                'date': trade_date,
                'earnings_date_next': earnings_date_next,
                'days_to_earnings': days_to,
                'earnings_surprise_last': earnings_surprise_last,
                'earnings_surprise_avg_4q': earnings_surprise_avg_4q,
                'revenue_surprise_last': revenue_surprise_last,
                'revenue_surprise_avg_4q': revenue_surprise_avg_4q,
                'beat_streak': beat_streak,
            }
            features.append(row)

        return pd.DataFrame(features)

    def save_features(self, symbol: str, features: pd.DataFrame) -> int:
        """Save features to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM features_earnings WHERE symbol = %s", (symbol,))

            columns = ['symbol', 'date', 'earnings_date_next', 'days_to_earnings',
                       'earnings_surprise_last', 'earnings_surprise_avg_4q',
                       'revenue_surprise_last', 'revenue_surprise_avg_4q', 'beat_streak']

            values = []
            for _, row in features.iterrows():
                record = [symbol]
                for col in columns[1:]:
                    val = row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        record.append(None)
                    elif isinstance(val, (np.floating, float)):
                        record.append(float(val) if not np.isnan(val) else None)
                    elif isinstance(val, (np.integer, int)):
                        record.append(int(val))
                    elif hasattr(val, 'date'):  # datetime
                        record.append(val)
                    else:
                        record.append(val)
                values.append(tuple(record))

            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO features_earnings ({', '.join(columns)}) VALUES ({placeholders})"
            cur.executemany(query, values)
            conn.commit()
            return len(values)
        finally:
            conn.close()

    def process_symbol(self, symbol: str) -> bool:
        """Process single symbol."""
        try:
            features = self.calculate_features(symbol)
            if features.empty:
                logger.warning(f"{symbol}: No features")
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
    parser = argparse.ArgumentParser(description='Calculate earnings features')
    parser.add_argument('--symbol', type=str, help='Single symbol')
    parser.add_argument('--all', action='store_true', help='All symbols')
    parser.add_argument('--limit', type=int, help='Limit symbols')
    parser.add_argument('--test', action='store_true', help='Test with AAPL')

    args = parser.parse_args()
    calc = EarningsCalculator()

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

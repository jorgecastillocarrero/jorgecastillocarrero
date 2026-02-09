"""
Sentiment Features Calculator
Combines AAII, CNN Fear & Greed, and M2 data into daily features.

Usage:
    python -m src.sentiment_calculator
    python -m src.sentiment_calculator --start 2020-01-01
"""

import logging
import argparse
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FMP_DATABASE_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"


class SentimentCalculator:
    """Calculate daily sentiment features from various sources."""

    def __init__(self, db_url: str = FMP_DATABASE_URL):
        self.db_url = db_url

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def get_trading_dates(self, start_date: Optional[date] = None) -> list:
        """Get all trading dates from price history."""
        conn = self.get_connection()
        try:
            query = """
                SELECT DISTINCT date FROM fmp_price_history
                WHERE symbol = 'SPY'
            """
            if start_date:
                query += f" AND date >= '{start_date}'"
            query += " ORDER BY date"

            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_aaii_data(self) -> pd.DataFrame:
        """Get AAII sentiment data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT date, bullish, neutral, bearish, bull_bear_spread
                FROM sentiment_aaii
                ORDER BY date
            """
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_fear_greed_data(self) -> pd.DataFrame:
        """Get CNN Fear & Greed data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT date, value, rating
                FROM sentiment_cnn_fear_greed
                ORDER BY date
            """
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_m2_data(self) -> pd.DataFrame:
        """Get M2 liquidity data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT date, m2_usa, m2_yoy_change
                FROM macro_global_m2
                ORDER BY date
            """
            df = pd.read_sql(query, conn)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def classify_aaii_zone(self, bull_bear_spread: float) -> str:
        """Classify AAII sentiment into zones based on bull-bear spread."""
        if bull_bear_spread is None:
            return None
        if bull_bear_spread <= -20:
            return 'extreme_bearish'
        elif bull_bear_spread <= -5:
            return 'bearish'
        elif bull_bear_spread <= 10:
            return 'neutral'
        elif bull_bear_spread <= 25:
            return 'bullish'
        else:
            return 'extreme_bullish'

    def classify_m2_zone(self, yoy_change: float) -> str:
        """Classify M2 growth into zones."""
        if yoy_change is None:
            return None
        if yoy_change < 0:
            return 'contracting'
        elif yoy_change < 3:
            return 'slow'
        elif yoy_change < 7:
            return 'normal'
        elif yoy_change < 15:
            return 'expanding'
        else:
            return 'rapid'

    def calculate_features(self, start_date: Optional[date] = None) -> pd.DataFrame:
        """Calculate daily sentiment features."""
        trading_dates = self.get_trading_dates(start_date)
        if not trading_dates:
            logger.warning("No trading dates found")
            return pd.DataFrame()

        logger.info(f"Processing {len(trading_dates)} trading dates...")

        # Load source data
        aaii_df = self.get_aaii_data()
        fg_df = self.get_fear_greed_data()
        m2_df = self.get_m2_data()

        logger.info(f"AAII: {len(aaii_df)} records")
        logger.info(f"Fear & Greed: {len(fg_df)} records")
        logger.info(f"M2: {len(m2_df)} records")

        features = []

        for trade_date in trading_dates:
            td = pd.Timestamp(trade_date)
            row = {'date': trade_date}

            # AAII (forward fill from weekly)
            aaii_past = aaii_df[aaii_df.index <= td]
            if not aaii_past.empty:
                latest_aaii = aaii_past.iloc[-1]
                row['aaii_bullish'] = latest_aaii['bullish']
                row['aaii_neutral'] = latest_aaii['neutral']
                row['aaii_bearish'] = latest_aaii['bearish']
                row['aaii_bull_bear_spread'] = latest_aaii['bull_bear_spread']
                row['aaii_zone'] = self.classify_aaii_zone(latest_aaii['bull_bear_spread'])

                # Change vs 4 weeks ago
                four_weeks_ago = td - timedelta(weeks=4)
                aaii_4w = aaii_df[aaii_df.index <= four_weeks_ago]
                if not aaii_4w.empty:
                    old_bullish = aaii_4w.iloc[-1]['bullish']
                    if old_bullish and old_bullish > 0:
                        row['aaii_bullish_change_4w'] = latest_aaii['bullish'] - old_bullish

            # CNN Fear & Greed
            fg_past = fg_df[fg_df.index <= td]
            if not fg_past.empty:
                latest_fg = fg_past.iloc[-1]
                row['fear_greed'] = latest_fg['value']
                row['fear_greed_zone'] = latest_fg['rating']

                # Value 1 week ago
                one_week_ago = td - timedelta(weeks=1)
                fg_1w = fg_df[fg_df.index <= one_week_ago]
                if not fg_1w.empty:
                    row['fear_greed_prev_1w'] = fg_1w.iloc[-1]['value']
                    if row['fear_greed_prev_1w'] and row['fear_greed_prev_1w'] > 0:
                        row['fear_greed_change_1w'] = row['fear_greed'] - row['fear_greed_prev_1w']

                # Value 1 month ago
                one_month_ago = td - timedelta(days=30)
                fg_1m = fg_df[fg_df.index <= one_month_ago]
                if not fg_1m.empty:
                    row['fear_greed_prev_1m'] = fg_1m.iloc[-1]['value']
                    if row['fear_greed_prev_1m'] and row['fear_greed_prev_1m'] > 0:
                        row['fear_greed_change_1m'] = row['fear_greed'] - row['fear_greed_prev_1m']

            # M2 (forward fill from monthly)
            m2_past = m2_df[m2_df.index <= td]
            if not m2_past.empty:
                latest_m2 = m2_past.iloc[-1]
                row['m2_usa'] = latest_m2['m2_usa']
                row['m2_yoy_change'] = latest_m2['m2_yoy_change']
                row['m2_zone'] = self.classify_m2_zone(latest_m2['m2_yoy_change'])

            features.append(row)

        return pd.DataFrame(features)

    def save_features(self, features: pd.DataFrame) -> int:
        """Save features to database."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Delete existing data for dates we're updating
            dates = features['date'].tolist()
            if dates:
                cur.execute("DELETE FROM features_sentiment WHERE date >= %s AND date <= %s",
                           (min(dates), max(dates)))

            columns = ['date', 'aaii_bullish', 'aaii_neutral', 'aaii_bearish',
                      'aaii_bull_bear_spread', 'aaii_bullish_change_4w', 'aaii_zone',
                      'fear_greed', 'fear_greed_zone', 'fear_greed_prev_1w',
                      'fear_greed_prev_1m', 'fear_greed_change_1w', 'fear_greed_change_1m',
                      'm2_usa', 'm2_yoy_change', 'm2_zone']

            count = 0
            for _, row in features.iterrows():
                values = []
                for col in columns:
                    val = row.get(col)
                    if pd.isna(val):
                        values.append(None)
                    else:
                        values.append(val)

                placeholders = ', '.join(['%s'] * len(columns))
                query = f"INSERT INTO features_sentiment ({', '.join(columns)}) VALUES ({placeholders})"
                cur.execute(query, values)
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()

    def process(self, start_date: Optional[date] = None):
        """Process and save sentiment features."""
        features = self.calculate_features(start_date)
        if features.empty:
            logger.warning("No features calculated")
            return 0

        count = self.save_features(features)
        logger.info(f"Saved {count} sentiment feature records")
        return count


def main():
    parser = argparse.ArgumentParser(description='Calculate sentiment features')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')

    args = parser.parse_args()

    start_date = None
    if args.start:
        start_date = date.fromisoformat(args.start)

    calc = SentimentCalculator()
    calc.process(start_date)


if __name__ == '__main__':
    main()

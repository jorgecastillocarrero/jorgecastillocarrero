"""
Aggregate Sentiment Calculator.
Combines news and transcript sentiment into daily features.
"""

import logging
import argparse
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

from ..config import get_nlp_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AggregateSentimentCalculator:
    """
    Calculator for aggregate daily sentiment features.

    Combines:
    - News sentiment (daily)
    - Transcript sentiment (forward-filled)
    - Macro sentiment (from existing features_sentiment)

    Saves to features_sentiment_daily table.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize calculator.

        Args:
            db_url: Database URL (defaults to FMP database)
        """
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def get_trading_dates(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[date]:
        """
        Get all trading dates from price history.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of trading dates
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT DISTINCT date FROM fmp_price_history
                WHERE symbol = 'SPY'
            """

            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"

            query += " ORDER BY date"

            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

        finally:
            conn.close()

    def get_symbols_with_sentiment(self) -> List[str]:
        """Get symbols that have sentiment data."""
        conn = self.get_connection()
        try:
            query = """
                SELECT DISTINCT symbol FROM nlp_sentiment_news
                UNION
                SELECT DISTINCT symbol FROM nlp_sentiment_transcript
            """

            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

        finally:
            conn.close()

    def get_news_sentiment(
        self,
        symbol: str,
        start_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get daily aggregated news sentiment.

        Args:
            symbol: Stock symbol
            start_date: Start date

        Returns:
            DataFrame with daily sentiment
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT
                    published_date as date,
                    AVG(ensemble_score) as news_sentiment,
                    COUNT(*) as news_count,
                    AVG(confidence) as news_confidence
                FROM nlp_sentiment_news
                WHERE symbol = %s
            """
            params = [symbol]

            if start_date:
                query += " AND published_date >= %s"
                params.append(start_date)

            query += " GROUP BY published_date ORDER BY published_date"

            return pd.read_sql(query, conn, params=params)

        finally:
            conn.close()

    def get_transcript_sentiment(self, symbol: str) -> pd.DataFrame:
        """
        Get transcript sentiment history.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with transcript sentiment
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT
                    earnings_date as date,
                    overall_score as transcript_sentiment,
                    qa_prepared_delta,
                    guidance_score
                FROM nlp_sentiment_transcript
                WHERE symbol = %s
                ORDER BY earnings_date
            """

            return pd.read_sql(query, conn, params=[symbol])

        finally:
            conn.close()

    def get_macro_sentiment(
        self,
        start_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get macro sentiment from existing features.

        Args:
            start_date: Start date

        Returns:
            DataFrame with macro sentiment
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT
                    date,
                    fear_greed,
                    aaii_bull_bear_spread as aaii_spread
                FROM features_sentiment
            """

            if start_date:
                query += f" WHERE date >= '{start_date}'"

            query += " ORDER BY date"

            return pd.read_sql(query, conn)

        finally:
            conn.close()

    def calculate_features(
        self,
        symbol: str,
        start_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Calculate all sentiment features for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date

        Returns:
            DataFrame with features
        """
        # Get trading dates
        trading_dates = self.get_trading_dates(start_date)
        if not trading_dates:
            return pd.DataFrame()

        # Create base DataFrame
        df = pd.DataFrame({'date': trading_dates})
        df['symbol'] = symbol

        # Get news sentiment
        news_df = self.get_news_sentiment(symbol, start_date)
        if not news_df.empty:
            news_df['date'] = pd.to_datetime(news_df['date']).dt.date
            df = df.merge(news_df, on='date', how='left')

            # Calculate moving averages
            df['news_sentiment_ma7'] = df['news_sentiment'].rolling(7, min_periods=1).mean()
            df['news_sentiment_momentum'] = df['news_sentiment'] - df['news_sentiment'].shift(7)
        else:
            df['news_sentiment'] = None
            df['news_count'] = 0
            df['news_sentiment_ma7'] = None
            df['news_sentiment_momentum'] = None

        # Get transcript sentiment
        transcript_df = self.get_transcript_sentiment(symbol)
        if not transcript_df.empty:
            transcript_df['date'] = pd.to_datetime(transcript_df['date']).dt.date

            # Forward fill transcript sentiment to trading dates
            df = df.merge(
                transcript_df[['date', 'transcript_sentiment']],
                on='date',
                how='left'
            )
            df['transcript_sentiment'] = df['transcript_sentiment'].ffill()

            # Calculate days since earnings
            earnings_dates = set(transcript_df['date'].tolist())

            def days_since(row_date):
                past_earnings = [d for d in earnings_dates if d <= row_date]
                if past_earnings:
                    return (row_date - max(past_earnings)).days
                return None

            df['days_since_earnings'] = df['date'].apply(days_since)
        else:
            df['transcript_sentiment'] = None
            df['days_since_earnings'] = None

        # Get macro sentiment
        macro_df = self.get_macro_sentiment(start_date)
        if not macro_df.empty:
            macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
            df = df.merge(macro_df, on='date', how='left')
        else:
            df['fear_greed'] = None
            df['aaii_spread'] = None

        # Calculate combined sentiment
        df['combined_sentiment'] = self._calculate_combined_sentiment(df)
        df['sentiment_zone'] = df['combined_sentiment'].apply(self._classify_zone)

        return df

    def _calculate_combined_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate combined sentiment score.

        Weights:
        - News sentiment: 40%
        - Transcript sentiment: 35%
        - Fear & Greed (normalized): 25%
        """
        combined = pd.Series(index=df.index, dtype=float)

        for idx, row in df.iterrows():
            scores = []
            weights = []

            # News sentiment
            if pd.notna(row.get('news_sentiment')):
                scores.append(row['news_sentiment'])
                weights.append(0.4)

            # Transcript sentiment
            if pd.notna(row.get('transcript_sentiment')):
                scores.append(row['transcript_sentiment'])
                weights.append(0.35)

            # Fear & Greed (normalize 0-100 to -1 to 1)
            if pd.notna(row.get('fear_greed')):
                fg_normalized = (row['fear_greed'] - 50) / 50
                scores.append(fg_normalized)
                weights.append(0.25)

            if scores:
                total_weight = sum(weights)
                combined[idx] = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                combined[idx] = None

        return combined

    def _classify_zone(self, score: Optional[float]) -> Optional[str]:
        """Classify sentiment score into zone."""
        if score is None or pd.isna(score):
            return None

        if score <= -0.5:
            return 'extreme_bearish'
        elif score <= -0.2:
            return 'bearish'
        elif score <= 0.2:
            return 'neutral'
        elif score <= 0.5:
            return 'bullish'
        else:
            return 'extreme_bullish'

    def save_features(
        self,
        features_df: pd.DataFrame
    ) -> int:
        """
        Save features to database.

        Args:
            features_df: DataFrame with features

        Returns:
            Number of records saved
        """
        if features_df.empty:
            return 0

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Delete existing records for this symbol and date range
            symbol = features_df['symbol'].iloc[0]
            min_date = features_df['date'].min()
            max_date = features_df['date'].max()

            cur.execute("""
                DELETE FROM features_sentiment_daily
                WHERE symbol = %s AND date >= %s AND date <= %s
            """, [symbol, min_date, max_date])

            # Prepare records
            columns = [
                'symbol', 'date', 'news_sentiment', 'news_count',
                'news_sentiment_ma7', 'news_sentiment_momentum',
                'transcript_sentiment', 'days_since_earnings',
                'fear_greed', 'aaii_spread', 'combined_sentiment', 'sentiment_zone'
            ]

            records = []
            for _, row in features_df.iterrows():
                record = []
                for col in columns:
                    val = row.get(col)
                    if pd.isna(val):
                        record.append(None)
                    else:
                        record.append(val)
                records.append(tuple(record))

            # Insert
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"""
                INSERT INTO features_sentiment_daily ({', '.join(columns)})
                VALUES ({placeholders})
            """

            execute_batch(cur, query, records, page_size=1000)
            conn.commit()

            logger.info(f"Saved {len(records)} sentiment features for {symbol}")
            return len(records)

        except Exception as e:
            logger.error(f"Error saving features: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def process_symbol(
        self,
        symbol: str,
        start_date: Optional[date] = None
    ) -> int:
        """
        Process features for a single symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date

        Returns:
            Number of records saved
        """
        logger.info(f"Processing sentiment features for {symbol}")

        features_df = self.calculate_features(symbol, start_date)
        if features_df.empty:
            return 0

        return self.save_features(features_df)

    def process(
        self,
        start_date: Optional[date] = None,
        symbols: Optional[List[str]] = None
    ) -> int:
        """
        Main processing method.

        Args:
            start_date: Start date
            symbols: List of symbols (None for all)

        Returns:
            Total records processed
        """
        if symbols is None:
            symbols = self.get_symbols_with_sentiment()

        logger.info(f"Processing aggregate sentiment for {len(symbols)} symbols")

        total = 0
        for i, symbol in enumerate(symbols):
            count = self.process_symbol(symbol, start_date)
            total += count

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(symbols)} symbols")

        logger.info(f"Total records saved: {total}")
        return total


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Calculate aggregate sentiment features')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, help='Process specific symbol')

    args = parser.parse_args()

    start_date = None
    if args.start:
        start_date = date.fromisoformat(args.start)

    calc = AggregateSentimentCalculator()

    if args.symbol:
        count = calc.process_symbol(args.symbol, start_date)
    else:
        count = calc.process(start_date)

    logger.info(f"Processed {count} records")


if __name__ == '__main__':
    main()

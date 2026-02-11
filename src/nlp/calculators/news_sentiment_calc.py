"""
News Sentiment Calculator.
Processes news articles and saves sentiment to database.
"""

import logging
import argparse
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

from ..config import get_nlp_settings
from ..analyzers.news_analyzer import NewsAnalyzer, NewsAnalysisResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsSentimentCalculator:
    """
    Calculator for news sentiment features.

    Processes news articles from news_history table,
    analyzes sentiment, and saves to nlp_sentiment_news.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize calculator.

        Args:
            db_url: Database URL (defaults to FMP database)
        """
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self._analyzer = None

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    @property
    def analyzer(self) -> NewsAnalyzer:
        """Lazy load analyzer."""
        if self._analyzer is None:
            self._analyzer = NewsAnalyzer()
        return self._analyzer

    def get_unprocessed_news(
        self,
        start_date: Optional[date] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get news articles that haven't been processed yet.

        Args:
            start_date: Start date for news
            limit: Maximum number of articles

        Returns:
            DataFrame with news articles
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT n.id, n.title, n.text as content, n.symbol,
                       n.published_utc as published_date, n.source
                FROM news_history n
                LEFT JOIN nlp_sentiment_news s ON n.id = s.news_id
                WHERE s.id IS NULL
            """

            if start_date:
                query += f" AND n.published_utc >= '{start_date}'"

            query += f" ORDER BY n.published_utc DESC LIMIT {limit}"

            df = pd.read_sql(query, conn)
            return df

        except Exception as e:
            logger.error(f"Error fetching unprocessed news: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_news_by_symbol(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get news for a specific symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with news articles
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT id, title, text as content, symbol,
                       published_utc as published_date, source
                FROM news_history
                WHERE symbol = %s
            """
            params = [symbol]

            if start_date:
                query += " AND published_utc >= %s"
                params.append(start_date)
            if end_date:
                query += " AND published_utc <= %s"
                params.append(end_date)

            query += " ORDER BY published_utc DESC"

            df = pd.read_sql(query, conn, params=params)
            return df

        finally:
            conn.close()

    def process_news(
        self,
        news_df: pd.DataFrame,
        batch_size: int = 32
    ) -> List[NewsAnalysisResult]:
        """
        Process news articles through analyzer.

        Args:
            news_df: DataFrame with news articles
            batch_size: Batch size for processing

        Returns:
            List of analysis results
        """
        if news_df.empty:
            return []

        # Convert to list of dicts for batch processing
        articles = []
        for _, row in news_df.iterrows():
            articles.append({
                'id': row.get('id'),
                'title': row.get('title', ''),
                'content': row.get('content', ''),
                'symbol': row.get('symbol', ''),
                'published_date': row.get('published_date'),
                'source': row.get('source', '')
            })

        # Process in batches
        results = self.analyzer.analyze_batch(articles, batch_size)

        # Add symbol from original data
        for i, result in enumerate(results):
            if i < len(articles):
                result.tickers = [articles[i].get('symbol', '')] + result.tickers

        return results

    def save_results(self, results: List[NewsAnalysisResult]) -> int:
        """
        Save analysis results to database.

        Args:
            results: List of NewsAnalysisResult

        Returns:
            Number of records saved
        """
        if not results:
            return 0

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Prepare data
            records = []
            for r in results:
                # Get primary symbol
                symbol = r.tickers[0] if r.tickers else ''
                published_date = r.published_date
                if isinstance(published_date, datetime):
                    published_date = published_date.date()

                records.append((
                    r.news_id,
                    symbol,
                    published_date,
                    r.finbert_score,
                    r.roberta_score,
                    r.ensemble_score,
                    r.ensemble_label,
                    r.confidence,
                    r.model_version
                ))

            # Insert
            query = """
                INSERT INTO nlp_sentiment_news
                (news_id, symbol, published_date, finbert_score, roberta_score,
                 ensemble_score, ensemble_label, confidence, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (news_id) DO UPDATE SET
                    finbert_score = EXCLUDED.finbert_score,
                    roberta_score = EXCLUDED.roberta_score,
                    ensemble_score = EXCLUDED.ensemble_score,
                    ensemble_label = EXCLUDED.ensemble_label,
                    confidence = EXCLUDED.confidence,
                    model_version = EXCLUDED.model_version,
                    processed_at = NOW()
            """

            execute_batch(cur, query, records, page_size=100)
            conn.commit()

            logger.info(f"Saved {len(records)} news sentiment records")
            return len(records)

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def calculate_daily_aggregates(
        self,
        symbol: str,
        start_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Calculate daily sentiment aggregates for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date

        Returns:
            DataFrame with daily aggregates
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT
                    published_date as date,
                    AVG(ensemble_score) as news_sentiment,
                    COUNT(*) as news_count,
                    AVG(confidence) as avg_confidence
                FROM nlp_sentiment_news
                WHERE symbol = %s
            """
            params = [symbol]

            if start_date:
                query += " AND published_date >= %s"
                params.append(start_date)

            query += " GROUP BY published_date ORDER BY published_date"

            df = pd.read_sql(query, conn, params=params)

            # Calculate moving averages
            if not df.empty:
                df['news_sentiment_ma7'] = df['news_sentiment'].rolling(7, min_periods=1).mean()
                df['news_sentiment_momentum'] = df['news_sentiment'] - df['news_sentiment'].shift(7)

            return df

        finally:
            conn.close()

    def process(
        self,
        start_date: Optional[date] = None,
        limit: int = 1000,
        batch_size: int = 32
    ) -> int:
        """
        Main processing method.

        Args:
            start_date: Start date for processing
            limit: Maximum number of articles
            batch_size: Batch size

        Returns:
            Number of records processed
        """
        logger.info(f"Processing news sentiment (limit={limit})")

        # Get unprocessed news
        news_df = self.get_unprocessed_news(start_date, limit)
        logger.info(f"Found {len(news_df)} unprocessed news articles")

        if news_df.empty:
            return 0

        # Process
        results = self.process_news(news_df, batch_size)

        # Save
        count = self.save_results(results)

        return count


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Calculate news sentiment')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum articles')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    start_date = None
    if args.start:
        start_date = date.fromisoformat(args.start)

    calc = NewsSentimentCalculator()
    count = calc.process(start_date, args.limit, args.batch_size)
    logger.info(f"Processed {count} articles")


if __name__ == '__main__':
    main()

"""
Incremental Processor.
Processes new data as it arrives for real-time sentiment updates.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta

from ..config import get_nlp_settings
from ..services.sentiment_service import get_sentiment_service
from ..analyzers.news_analyzer import NewsAnalyzer
from ..storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class IncrementalProcessor:
    """
    Incremental processing for real-time updates.

    Designed for:
    - Scheduled jobs (hourly, daily)
    - Event-driven processing
    - Low-latency updates
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize incremental processor.

        Args:
            db_url: Database URL
        """
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self._storage = None
        self._news_analyzer = None
        self._last_processed = {}

    @property
    def storage(self) -> PostgresStorage:
        """Lazy load storage."""
        if self._storage is None:
            self._storage = PostgresStorage(self.db_url)
        return self._storage

    @property
    def news_analyzer(self) -> NewsAnalyzer:
        """Lazy load news analyzer."""
        if self._news_analyzer is None:
            self._news_analyzer = NewsAnalyzer()
        return self._news_analyzer

    def process_new_news(
        self,
        since_hours: int = 4,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Process news articles added in the last N hours.

        Args:
            since_hours: Hours to look back
            limit: Maximum articles

        Returns:
            Dict with processing results
        """
        logger.info(f"Processing news from last {since_hours} hours")

        results = {
            'source': 'news',
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

        try:
            # Get recent unprocessed news
            import psycopg2

            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()

            since_date = datetime.now() - timedelta(hours=since_hours)

            query = """
                SELECT n.id, n.title, n.text, n.symbol, n.published_utc, n.source
                FROM news_history n
                LEFT JOIN nlp_sentiment_news s ON n.id = s.news_id
                WHERE s.id IS NULL
                    AND n.published_utc >= %s
                ORDER BY n.published_utc DESC
                LIMIT %s
            """

            cur.execute(query, [since_date, limit])
            rows = cur.fetchall()
            conn.close()

            if not rows:
                logger.info("No new news to process")
                return results

            # Process each article
            for row in rows:
                news_id, title, content, symbol, published, source = row

                try:
                    analysis = self.news_analyzer.analyze(
                        title=title or '',
                        content=content or '',
                        news_id=news_id,
                        published_date=published,
                        source=source or ''
                    )

                    # Save result
                    self.storage.save_sentiment_results(
                        [analysis.to_dict()],
                        source_type='news'
                    )

                    results['processed'] += 1

                except Exception as e:
                    logger.error(f"Error processing news {news_id}: {e}")
                    results['failed'] += 1

            logger.info(f"Processed {results['processed']} news articles")

        except Exception as e:
            logger.error(f"Error in incremental news processing: {e}")

        return results

    def process_symbol_news(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Process recent news for a specific symbol.

        Args:
            symbol: Stock symbol
            days: Days to look back

        Returns:
            Dict with processing results
        """
        logger.info(f"Processing news for {symbol} (last {days} days)")

        results = {
            'symbol': symbol,
            'processed': 0,
            'failed': 0
        }

        try:
            import psycopg2

            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()

            since_date = date.today() - timedelta(days=days)

            query = """
                SELECT n.id, n.title, n.text, n.published_utc, n.source
                FROM news_history n
                LEFT JOIN nlp_sentiment_news s ON n.id = s.news_id
                WHERE s.id IS NULL
                    AND n.symbol = %s
                    AND n.published_utc >= %s
                ORDER BY n.published_utc DESC
            """

            cur.execute(query, [symbol, since_date])
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                news_id, title, content, published, source = row

                try:
                    analysis = self.news_analyzer.analyze(
                        title=title or '',
                        content=content or '',
                        news_id=news_id,
                        published_date=published,
                        source=source or ''
                    )

                    self.storage.save_sentiment_results(
                        [analysis.to_dict()],
                        source_type='news'
                    )
                    results['processed'] += 1

                except Exception as e:
                    logger.error(f"Error processing news {news_id}: {e}")
                    results['failed'] += 1

            logger.info(f"Processed {results['processed']} news for {symbol}")

        except Exception as e:
            logger.error(f"Error processing symbol news: {e}")

        return results

    def update_daily_features(
        self,
        symbol: str,
        target_date: Optional[date] = None
    ) -> bool:
        """
        Update daily sentiment features for a symbol.

        Args:
            symbol: Stock symbol
            target_date: Date to update (default: today)

        Returns:
            True if successful
        """
        from ..calculators.aggregate_sentiment_calc import AggregateSentimentCalculator

        target_date = target_date or date.today()
        logger.info(f"Updating daily features for {symbol} on {target_date}")

        try:
            calc = AggregateSentimentCalculator(self.db_url)

            # Calculate features for just this date range
            start = target_date - timedelta(days=7)  # Need some history for MAs
            features_df = calc.calculate_features(symbol, start)

            if features_df.empty:
                return False

            # Filter to target date only for saving
            features_df = features_df[features_df['date'] == target_date]

            if features_df.empty:
                return False

            calc.save_features(features_df)
            return True

        except Exception as e:
            logger.error(f"Error updating daily features: {e}")
            return False

    def get_current_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current sentiment for a symbol.

        Combines:
        - Latest news sentiment
        - Most recent transcript sentiment
        - Macro indicators

        Args:
            symbol: Stock symbol

        Returns:
            Dict with current sentiment or None
        """
        try:
            import psycopg2

            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }

            # Latest news sentiment (last 7 days)
            cur.execute("""
                SELECT AVG(ensemble_score), COUNT(*), MAX(published_date)
                FROM nlp_sentiment_news
                WHERE symbol = %s
                    AND published_date >= CURRENT_DATE - INTERVAL '7 days'
            """, [symbol])
            row = cur.fetchone()
            result['news_sentiment'] = row[0]
            result['news_count_7d'] = row[1]
            result['latest_news_date'] = row[2]

            # Latest transcript sentiment
            cur.execute("""
                SELECT overall_score, qa_prepared_delta, year, quarter
                FROM nlp_sentiment_transcript
                WHERE symbol = %s
                ORDER BY earnings_date DESC
                LIMIT 1
            """, [symbol])
            row = cur.fetchone()
            if row:
                result['transcript_sentiment'] = row[0]
                result['qa_delta'] = row[1]
                result['latest_earnings'] = f"{row[2]} {row[3]}"

            # Latest macro
            cur.execute("""
                SELECT fear_greed, aaii_bull_bear_spread
                FROM features_sentiment
                ORDER BY date DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                result['fear_greed'] = row[0]
                result['aaii_spread'] = row[1]

            conn.close()

            # Calculate combined sentiment
            scores = []
            if result.get('news_sentiment') is not None:
                scores.append(result['news_sentiment'])
            if result.get('transcript_sentiment') is not None:
                scores.append(result['transcript_sentiment'])

            if scores:
                result['combined_sentiment'] = sum(scores) / len(scores)
            else:
                result['combined_sentiment'] = None

            return result

        except Exception as e:
            logger.error(f"Error getting current sentiment: {e}")
            return None

    def run_scheduled_update(self) -> Dict[str, Any]:
        """
        Run scheduled incremental update.

        Designed to be called by scheduler every 4 hours.

        Returns:
            Dict with update results
        """
        logger.info("Running scheduled incremental update")

        results = {
            'timestamp': datetime.now().isoformat(),
            'news': None,
            'features_updated': 0
        }

        # Process new news
        results['news'] = self.process_new_news(since_hours=4, limit=200)

        # Update daily features for active symbols
        try:
            import psycopg2

            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()

            # Get symbols with recent news
            cur.execute("""
                SELECT DISTINCT symbol
                FROM nlp_sentiment_news
                WHERE processed_at >= CURRENT_TIMESTAMP - INTERVAL '4 hours'
            """)
            symbols = [row[0] for row in cur.fetchall()]
            conn.close()

            for symbol in symbols:
                if self.update_daily_features(symbol):
                    results['features_updated'] += 1

        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")

        logger.info(f"Scheduled update complete: {results}")
        return results


def run_incremental_update(since_hours: int = 4) -> Dict[str, Any]:
    """
    Convenience function to run incremental update.

    Args:
        since_hours: Hours to look back

    Returns:
        Dict with update results
    """
    processor = IncrementalProcessor()
    return processor.run_scheduled_update()

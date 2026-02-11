"""
Transcript Sentiment Calculator.
Processes earnings call transcripts and saves sentiment to database.
"""

import logging
import argparse
from datetime import date, datetime
from typing import Optional, List, Dict
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch, Json

from ..config import get_nlp_settings
from ..analyzers.transcript_analyzer import TranscriptAnalyzer, TranscriptAnalysisResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TranscriptSentimentCalculator:
    """
    Calculator for transcript sentiment features.

    Processes earnings call transcripts from fmp_earnings_transcripts,
    analyzes sentiment, and saves to nlp_sentiment_transcript.
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
    def analyzer(self) -> TranscriptAnalyzer:
        """Lazy load analyzer."""
        if self._analyzer is None:
            self._analyzer = TranscriptAnalyzer()
        return self._analyzer

    def get_unprocessed_transcripts(
        self,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get transcripts that haven't been processed yet.

        Args:
            limit: Maximum number of transcripts

        Returns:
            DataFrame with transcripts
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT t.symbol, t.year, t.quarter, t.content, t.date
                FROM fmp_earnings_transcripts t
                LEFT JOIN nlp_sentiment_transcript s
                    ON t.symbol = s.symbol AND t.year = s.year AND t.quarter = s.quarter
                WHERE s.id IS NULL
                    AND t.content IS NOT NULL
                    AND LENGTH(t.content) > 100
                ORDER BY t.date DESC
                LIMIT %s
            """

            df = pd.read_sql(query, conn, params=[limit])
            return df

        except Exception as e:
            logger.error(f"Error fetching unprocessed transcripts: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_transcripts_by_symbol(
        self,
        symbol: str,
        years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Get transcripts for a specific symbol.

        Args:
            symbol: Stock symbol
            years: Optional list of years to filter

        Returns:
            DataFrame with transcripts
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT symbol, year, quarter, content, date
                FROM fmp_earnings_transcripts
                WHERE symbol = %s
                    AND content IS NOT NULL
            """
            params = [symbol]

            if years:
                query += f" AND year IN ({','.join(['%s']*len(years))})"
                params.extend(years)

            query += " ORDER BY date DESC"

            df = pd.read_sql(query, conn, params=params)
            return df

        finally:
            conn.close()

    def process_transcripts(
        self,
        transcripts_df: pd.DataFrame
    ) -> List[TranscriptAnalysisResult]:
        """
        Process transcripts through analyzer.

        Args:
            transcripts_df: DataFrame with transcripts

        Returns:
            List of analysis results
        """
        if transcripts_df.empty:
            return []

        results = []

        for _, row in transcripts_df.iterrows():
            try:
                result = self.analyzer.analyze(
                    transcript=row.get('content', ''),
                    symbol=row.get('symbol', ''),
                    year=int(row.get('year', 0)),
                    quarter=row.get('quarter', ''),
                    earnings_date=row.get('date')
                )
                results.append(result)

                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)} transcripts...")

            except Exception as e:
                logger.error(f"Error processing transcript {row.get('symbol')} "
                           f"{row.get('year')} {row.get('quarter')}: {e}")

        return results

    def save_results(self, results: List[TranscriptAnalysisResult]) -> int:
        """
        Save analysis results to database.

        Args:
            results: List of TranscriptAnalysisResult

        Returns:
            Number of records saved
        """
        if not results:
            return 0

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            records = []
            for r in results:
                earnings_date = r.earnings_date
                if isinstance(earnings_date, datetime):
                    earnings_date = earnings_date.date()

                records.append((
                    r.symbol,
                    r.year,
                    r.quarter,
                    earnings_date,
                    r.overall_score,
                    r.prepared_remarks_score,
                    r.qa_section_score,
                    r.guidance_score,
                    r.qa_prepared_delta,
                    Json(r.topics),
                    r.num_segments
                ))

            query = """
                INSERT INTO nlp_sentiment_transcript
                (symbol, year, quarter, earnings_date, overall_score,
                 prepared_remarks_score, qa_section_score, guidance_score,
                 qa_prepared_delta, topics, num_segments)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, year, quarter) DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    prepared_remarks_score = EXCLUDED.prepared_remarks_score,
                    qa_section_score = EXCLUDED.qa_section_score,
                    guidance_score = EXCLUDED.guidance_score,
                    qa_prepared_delta = EXCLUDED.qa_prepared_delta,
                    topics = EXCLUDED.topics,
                    num_segments = EXCLUDED.num_segments,
                    processed_at = NOW()
            """

            execute_batch(cur, query, records, page_size=50)
            conn.commit()

            logger.info(f"Saved {len(records)} transcript sentiment records")
            return len(records)

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_sentiment_history(
        self,
        symbol: str,
        lookback_quarters: int = 8
    ) -> pd.DataFrame:
        """
        Get sentiment history for a symbol.

        Args:
            symbol: Stock symbol
            lookback_quarters: Number of quarters to look back

        Returns:
            DataFrame with sentiment history
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT
                    symbol, year, quarter, earnings_date,
                    overall_score, prepared_remarks_score,
                    qa_section_score, guidance_score,
                    qa_prepared_delta, topics
                FROM nlp_sentiment_transcript
                WHERE symbol = %s
                ORDER BY year DESC, quarter DESC
                LIMIT %s
            """

            df = pd.read_sql(query, conn, params=[symbol, lookback_quarters])
            return df

        finally:
            conn.close()

    def get_latest_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Get most recent sentiment for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with latest sentiment or None
        """
        conn = self.get_connection()
        try:
            query = """
                SELECT *
                FROM nlp_sentiment_transcript
                WHERE symbol = %s
                ORDER BY earnings_date DESC
                LIMIT 1
            """

            cur = conn.cursor()
            cur.execute(query, [symbol])
            row = cur.fetchone()

            if row:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row))
            return None

        finally:
            conn.close()

    def process(self, limit: int = 100) -> int:
        """
        Main processing method.

        Args:
            limit: Maximum number of transcripts

        Returns:
            Number of records processed
        """
        logger.info(f"Processing transcript sentiment (limit={limit})")

        # Get unprocessed transcripts
        transcripts_df = self.get_unprocessed_transcripts(limit)
        logger.info(f"Found {len(transcripts_df)} unprocessed transcripts")

        if transcripts_df.empty:
            return 0

        # Process
        results = self.process_transcripts(transcripts_df)

        # Save
        count = self.save_results(results)

        return count


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Calculate transcript sentiment')
    parser.add_argument('--limit', type=int, default=100, help='Maximum transcripts')
    parser.add_argument('--symbol', type=str, help='Process specific symbol')

    args = parser.parse_args()

    calc = TranscriptSentimentCalculator()

    if args.symbol:
        # Process specific symbol
        transcripts_df = calc.get_transcripts_by_symbol(args.symbol)
        results = calc.process_transcripts(transcripts_df)
        count = calc.save_results(results)
    else:
        count = calc.process(args.limit)

    logger.info(f"Processed {count} transcripts")


if __name__ == '__main__':
    main()

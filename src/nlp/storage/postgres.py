"""
PostgreSQL Storage Implementation.
Implements BaseStorage for PostgreSQL with pgvector support.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch, Json

from .base import BaseStorage
from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class PostgresStorage(BaseStorage):
    """
    PostgreSQL implementation of storage backend.

    Features:
    - Partitioned tables for scalability
    - pgvector extension for embeddings
    - Efficient batch operations
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize PostgreSQL storage.

        Args:
            db_url: Database URL
        """
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self._has_pgvector = None

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def has_pgvector(self) -> bool:
        """Check if pgvector extension is available."""
        if self._has_pgvector is not None:
            return self._has_pgvector

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """)
            self._has_pgvector = cur.fetchone()[0]
            return self._has_pgvector
        except Exception:
            self._has_pgvector = False
            return False
        finally:
            conn.close()

    def save_sentiment_results(
        self,
        results: List[Dict[str, Any]],
        source_type: str
    ) -> int:
        """Save sentiment analysis results."""
        if not results:
            return 0

        if source_type == 'news':
            return self._save_news_sentiment(results)
        elif source_type == 'transcript':
            return self._save_transcript_sentiment(results)
        else:
            logger.error(f"Unknown source type: {source_type}")
            return 0

    def _save_news_sentiment(self, results: List[Dict]) -> int:
        """Save news sentiment results."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            records = []
            for r in results:
                records.append((
                    r.get('news_id'),
                    r.get('symbol', ''),
                    r.get('published_date'),
                    r.get('finbert_score'),
                    r.get('roberta_score'),
                    r.get('ensemble_score'),
                    r.get('ensemble_label', 'neutral'),
                    r.get('confidence', 0.0),
                    r.get('model_version', '')
                ))

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
            return len(records)

        except Exception as e:
            logger.error(f"Error saving news sentiment: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def _save_transcript_sentiment(self, results: List[Dict]) -> int:
        """Save transcript sentiment results."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            records = []
            for r in results:
                records.append((
                    r.get('symbol', ''),
                    r.get('year', 0),
                    r.get('quarter', ''),
                    r.get('earnings_date'),
                    r.get('overall_score', 0.0),
                    r.get('prepared_remarks_score'),
                    r.get('qa_section_score'),
                    r.get('guidance_score'),
                    r.get('qa_prepared_delta'),
                    Json(r.get('topics', [])),
                    r.get('num_segments', 0)
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
            return len(records)

        except Exception as e:
            logger.error(f"Error saving transcript sentiment: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def get_sentiment_for_symbol(
        self,
        symbol: str,
        source_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get sentiment results for a symbol."""
        conn = self.get_connection()
        try:
            if source_type == 'news':
                query = """
                    SELECT * FROM nlp_sentiment_news
                    WHERE symbol = %s
                """
                date_col = 'published_date'
            else:
                query = """
                    SELECT * FROM nlp_sentiment_transcript
                    WHERE symbol = %s
                """
                date_col = 'earnings_date'

            params = [symbol]

            if start_date:
                query += f" AND {date_col} >= %s"
                params.append(start_date)
            if end_date:
                query += f" AND {date_col} <= %s"
                params.append(end_date)

            query += f" ORDER BY {date_col} DESC LIMIT %s"
            params.append(limit)

            return pd.read_sql(query, conn, params=params)

        finally:
            conn.close()

    def get_daily_sentiment(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get daily aggregated sentiment."""
        conn = self.get_connection()
        try:
            query = """
                SELECT * FROM features_sentiment_daily
                WHERE symbol = %s
            """
            params = [symbol]

            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)

            query += " ORDER BY date"

            return pd.read_sql(query, conn, params=params)

        finally:
            conn.close()

    def save_embeddings(self, embeddings: List[Dict[str, Any]]) -> int:
        """Save text embeddings."""
        if not embeddings:
            return 0

        if not self.has_pgvector():
            logger.warning("pgvector not available, skipping embedding storage")
            return 0

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            records = []
            for e in embeddings:
                # Convert numpy array to list if needed
                vector = e.get('embedding')
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()

                records.append((
                    e.get('source_type', ''),
                    e.get('source_id'),
                    e.get('symbol', ''),
                    vector,
                    e.get('model_name', '')
                ))

            query = """
                INSERT INTO nlp_embeddings
                (source_type, source_id, symbol, embedding, model_name)
                VALUES (%s, %s, %s, %s, %s)
            """

            execute_batch(cur, query, records, page_size=100)
            conn.commit()
            return len(records)

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def search_similar(
        self,
        embedding: List[float],
        source_type: Optional[str] = None,
        symbol: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using pgvector."""
        if not self.has_pgvector():
            logger.warning("pgvector not available")
            return []

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Build query with optional filters
            query = """
                SELECT id, source_type, source_id, symbol,
                       1 - (embedding <=> %s::vector) as similarity
                FROM nlp_embeddings
                WHERE 1=1
            """
            params = [embedding]

            if source_type:
                query += " AND source_type = %s"
                params.append(source_type)
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)

            query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([embedding, k])

            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]

            results = []
            for row in cur.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
        finally:
            conn.close()

    def get_unprocessed_records(
        self,
        source_type: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get records that haven't been processed."""
        conn = self.get_connection()
        try:
            if source_type == 'news':
                query = """
                    SELECT n.id, n.title, n.text as content, n.symbol,
                           n.published_utc as published_date, n.source
                    FROM news_history n
                    LEFT JOIN nlp_sentiment_news s ON n.id = s.news_id
                    WHERE s.id IS NULL
                    ORDER BY n.published_utc DESC
                    LIMIT %s
                """
            else:
                query = """
                    SELECT t.symbol, t.year, t.quarter, t.content, t.date
                    FROM fmp_earnings_transcripts t
                    LEFT JOIN nlp_sentiment_transcript s
                        ON t.symbol = s.symbol AND t.year = s.year
                        AND t.quarter = s.quarter
                    WHERE s.id IS NULL
                        AND t.content IS NOT NULL
                    ORDER BY t.date DESC
                    LIMIT %s
                """

            return pd.read_sql(query, conn, params=[limit])

        finally:
            conn.close()

    def mark_as_processed(
        self,
        record_ids: List[int],
        source_type: str
    ) -> int:
        """Mark records as processed."""
        # Not needed as we insert into separate tables
        return len(record_ids)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            stats = {}

            # News stats
            cur.execute("SELECT COUNT(*) FROM nlp_sentiment_news")
            stats['news_processed'] = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM news_history n
                LEFT JOIN nlp_sentiment_news s ON n.id = s.news_id
                WHERE s.id IS NULL
            """)
            stats['news_pending'] = cur.fetchone()[0]

            # Transcript stats
            cur.execute("SELECT COUNT(*) FROM nlp_sentiment_transcript")
            stats['transcripts_processed'] = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM fmp_earnings_transcripts t
                LEFT JOIN nlp_sentiment_transcript s
                    ON t.symbol = s.symbol AND t.year = s.year AND t.quarter = s.quarter
                WHERE s.id IS NULL AND t.content IS NOT NULL
            """)
            stats['transcripts_pending'] = cur.fetchone()[0]

            # Embedding stats
            if self.has_pgvector():
                cur.execute("SELECT COUNT(*) FROM nlp_embeddings")
                stats['embeddings_stored'] = cur.fetchone()[0]
            else:
                stats['embeddings_stored'] = 0
                stats['pgvector_available'] = False

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
        finally:
            conn.close()

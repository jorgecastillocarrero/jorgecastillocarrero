"""
Financial Search Service.
Provides semantic search over earnings transcripts and company profiles.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import psycopg2

from ..config import get_nlp_settings
from .embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    source_type: str  # 'transcript' or 'profile'
    symbol: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.source_type}] {self.symbol} (score: {self.score:.3f})"


@dataclass
class SearchResponse:
    """Search response with multiple results."""
    query: str
    results: List[SearchResult]
    total_found: int

    def to_context(self, max_results: int = 5) -> str:
        """Convert results to context string for RAG."""
        if not self.results:
            return ""

        lines = []
        for r in self.results[:max_results]:
            lines.append(f"\n--- [{r.source_type.upper()}] {r.symbol} (relevance: {r.score:.2f}) ---")
            if r.metadata.get('year'):
                lines.append(f"Period: Q{r.metadata.get('quarter', '?')} {r.metadata['year']}")
            if r.metadata.get('section'):
                lines.append(f"Section: {r.metadata['section']}")
            lines.append(r.text[:1500])  # Limit text length

        return "\n".join(lines)


class FinancialSearchService:
    """
    Semantic search service for financial data.

    Provides:
    - Transcript search with year/quarter filtering
    - Company profile search with sector filtering
    - Combined search across all sources
    """

    def __init__(self, db_url: str = None):
        """Initialize search service."""
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self.embedding_service = get_embedding_service()
        self._has_pgvector = None

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def _check_pgvector(self) -> bool:
        """Check if pgvector is available."""
        if self._has_pgvector is not None:
            return self._has_pgvector

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            self._has_pgvector = cur.fetchone()[0]
            return self._has_pgvector
        except Exception:
            self._has_pgvector = False
            return False
        finally:
            conn.close()

    def is_available(self) -> bool:
        """Check if service is available."""
        return self._check_pgvector() and self.embedding_service.is_available()

    def search_transcripts(
        self,
        query: str,
        symbol: str = None,
        year: int = None,
        quarter: str = None,
        section: str = None,
        limit: int = 10
    ) -> SearchResponse:
        """
        Search earnings call transcripts.

        Args:
            query: Search query
            symbol: Filter by stock symbol
            year: Filter by year
            quarter: Filter by quarter (Q1, Q2, Q3, Q4)
            section: Filter by section (prepared_remarks, qa)
            limit: Maximum results

        Returns:
            SearchResponse with results
        """
        return self._search(
            query=query,
            source_type='transcript',
            symbol=symbol,
            year=year,
            quarter=quarter,
            section=section,
            limit=limit
        )

    def search_companies(
        self,
        query: str,
        sector: str = None,
        industry: str = None,
        country: str = None,
        limit: int = 10
    ) -> SearchResponse:
        """
        Search company profiles.

        Args:
            query: Search query
            sector: Filter by sector
            industry: Filter by industry
            country: Filter by country
            limit: Maximum results

        Returns:
            SearchResponse with results
        """
        return self._search(
            query=query,
            source_type='profile',
            sector=sector,
            industry=industry,
            country=country,
            limit=limit
        )

    def search_all(
        self,
        query: str,
        symbol: str = None,
        limit: int = 10
    ) -> SearchResponse:
        """
        Search across all sources (transcripts and profiles).

        Args:
            query: Search query
            symbol: Filter by stock symbol
            limit: Maximum results

        Returns:
            SearchResponse with results
        """
        return self._search(
            query=query,
            source_type=None,  # All types
            symbol=symbol,
            limit=limit
        )

    def _search(
        self,
        query: str,
        source_type: str = None,
        symbol: str = None,
        year: int = None,
        quarter: str = None,
        section: str = None,
        sector: str = None,
        industry: str = None,
        country: str = None,
        limit: int = 10
    ) -> SearchResponse:
        """Internal search method with all filters."""
        if not self.is_available():
            return SearchResponse(query=query, results=[], total_found=0)

        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return SearchResponse(query=query, results=[], total_found=0)

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Build query with filters
            sql = """
                SELECT
                    e.text_content,
                    e.source_type,
                    e.symbol,
                    1 - (e.embedding <=> %s::vector) as similarity,
                    e.year,
                    e.quarter,
                    e.section,
                    e.metadata
                FROM nlp_embeddings e
                WHERE 1=1
            """
            params = [query_embedding.tolist()]

            if source_type:
                sql += " AND e.source_type = %s"
                params.append(source_type)

            if symbol:
                sql += " AND e.symbol = %s"
                params.append(symbol.upper())

            if year:
                sql += " AND e.year = %s"
                params.append(year)

            if quarter:
                sql += " AND e.quarter = %s"
                params.append(quarter)

            if section:
                sql += " AND e.section = %s"
                params.append(section)

            # Metadata filters for profiles
            if sector:
                sql += " AND e.metadata->>'sector' ILIKE %s"
                params.append(f"%{sector}%")

            if industry:
                sql += " AND e.metadata->>'industry' ILIKE %s"
                params.append(f"%{industry}%")

            if country:
                sql += " AND e.metadata->>'country' ILIKE %s"
                params.append(f"%{country}%")

            sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding.tolist(), limit])

            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                text, src_type, sym, sim, yr, qtr, sec, meta = row
                results.append(SearchResult(
                    text=text or "",
                    source_type=src_type,
                    symbol=sym,
                    score=float(sim) if sim else 0,
                    metadata={
                        'year': yr,
                        'quarter': qtr,
                        'section': sec,
                        **(meta if isinstance(meta, dict) else {})
                    }
                ))

            return SearchResponse(
                query=query,
                results=results,
                total_found=len(results)
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResponse(query=query, results=[], total_found=0)
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed data."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            stats = {}

            # Total embeddings
            cur.execute("SELECT COUNT(*) FROM nlp_embeddings")
            stats['total_embeddings'] = cur.fetchone()[0]

            # By source type
            cur.execute("""
                SELECT source_type, COUNT(*)
                FROM nlp_embeddings
                GROUP BY source_type
            """)
            stats['by_source'] = dict(cur.fetchall())

            # Unique symbols
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM nlp_embeddings")
            stats['unique_symbols'] = cur.fetchone()[0]

            # pgvector available
            stats['pgvector_available'] = self._check_pgvector()

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
        finally:
            conn.close()


# Singleton instance
_search_service: Optional[FinancialSearchService] = None


def get_financial_search_service() -> FinancialSearchService:
    """Get the singleton search service instance."""
    global _search_service

    if _search_service is None:
        _search_service = FinancialSearchService()

    return _search_service

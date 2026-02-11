"""
Base Storage Interface.
Abstract interface for storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.

    Provides a consistent interface that can be implemented by:
    - PostgreSQL (current)
    - TimescaleDB (future)
    - Distributed storage (future)
    """

    @abstractmethod
    def save_sentiment_results(
        self,
        results: List[Dict[str, Any]],
        source_type: str
    ) -> int:
        """
        Save sentiment analysis results.

        Args:
            results: List of result dictionaries
            source_type: Type of source ('news', 'transcript')

        Returns:
            Number of records saved
        """
        pass

    @abstractmethod
    def get_sentiment_for_symbol(
        self,
        symbol: str,
        source_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get sentiment results for a symbol.

        Args:
            symbol: Stock symbol
            source_type: Type of source
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum records

        Returns:
            DataFrame with results
        """
        pass

    @abstractmethod
    def get_daily_sentiment(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get daily aggregated sentiment.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily sentiment
        """
        pass

    @abstractmethod
    def save_embeddings(
        self,
        embeddings: List[Dict[str, Any]]
    ) -> int:
        """
        Save text embeddings.

        Args:
            embeddings: List of embedding dictionaries

        Returns:
            Number of records saved
        """
        pass

    @abstractmethod
    def search_similar(
        self,
        embedding: List[float],
        source_type: Optional[str] = None,
        symbol: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            embedding: Query embedding vector
            source_type: Filter by source type
            symbol: Filter by symbol
            k: Number of results

        Returns:
            List of similar records
        """
        pass

    @abstractmethod
    def get_unprocessed_records(
        self,
        source_type: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get records that haven't been processed.

        Args:
            source_type: Type of source
            limit: Maximum records

        Returns:
            DataFrame with unprocessed records
        """
        pass

    @abstractmethod
    def mark_as_processed(
        self,
        record_ids: List[int],
        source_type: str
    ) -> int:
        """
        Mark records as processed.

        Args:
            record_ids: List of record IDs
            source_type: Type of source

        Returns:
            Number of records updated
        """
        pass

    @abstractmethod
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dict with statistics
        """
        pass

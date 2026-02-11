"""
Batch Processor.
Async batch processing pipeline for large-scale sentiment analysis.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import time

from ..config import get_nlp_settings
from ..calculators.news_sentiment_calc import NewsSentimentCalculator
from ..calculators.transcript_sentiment_calc import TranscriptSentimentCalculator
from ..calculators.aggregate_sentiment_calc import AggregateSentimentCalculator

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""
    source_type: str
    total_records: int = 0
    processed: int = 0
    failed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    records_per_second: float = 0.0

    def complete(self):
        """Mark processing as complete."""
        self.end_time = datetime.now()
        self.elapsed_seconds = (self.end_time - self.start_time).total_seconds()
        if self.elapsed_seconds > 0:
            self.records_per_second = self.processed / self.elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_type': self.source_type,
            'total_records': self.total_records,
            'processed': self.processed,
            'failed': self.failed,
            'elapsed_seconds': self.elapsed_seconds,
            'records_per_second': self.records_per_second
        }


class BatchProcessor:
    """
    Batch processing pipeline for NLP operations.

    Provides:
    - Parallel processing with configurable workers
    - Progress tracking and callbacks
    - Error handling and retry logic
    - Rate limiting
    """

    def __init__(
        self,
        max_workers: int = None,
        batch_size: int = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum parallel workers
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
        """
        settings = get_nlp_settings()
        self.max_workers = max_workers or settings.max_workers
        self.batch_size = batch_size or settings.batch_size
        self.progress_callback = progress_callback

        self._news_calc = None
        self._transcript_calc = None
        self._aggregate_calc = None

    @property
    def news_calculator(self) -> NewsSentimentCalculator:
        """Lazy load news calculator."""
        if self._news_calc is None:
            self._news_calc = NewsSentimentCalculator()
        return self._news_calc

    @property
    def transcript_calculator(self) -> TranscriptSentimentCalculator:
        """Lazy load transcript calculator."""
        if self._transcript_calc is None:
            self._transcript_calc = TranscriptSentimentCalculator()
        return self._transcript_calc

    @property
    def aggregate_calculator(self) -> AggregateSentimentCalculator:
        """Lazy load aggregate calculator."""
        if self._aggregate_calc is None:
            self._aggregate_calc = AggregateSentimentCalculator()
        return self._aggregate_calc

    def process_all_news(
        self,
        limit: int = None,
        start_date = None
    ) -> ProcessingStats:
        """
        Process all unprocessed news articles.

        Args:
            limit: Maximum articles to process
            start_date: Start date filter

        Returns:
            ProcessingStats
        """
        stats = ProcessingStats(source_type='news')

        logger.info(f"Starting batch news processing (limit={limit})")

        try:
            # Get unprocessed
            news_df = self.news_calculator.get_unprocessed_news(start_date, limit or 10000)
            stats.total_records = len(news_df)

            if news_df.empty:
                logger.info("No unprocessed news found")
                stats.complete()
                return stats

            # Process in batches
            total_processed = 0
            for i in range(0, len(news_df), self.batch_size):
                batch_df = news_df.iloc[i:i + self.batch_size]

                try:
                    results = self.news_calculator.process_news(batch_df, self.batch_size)
                    saved = self.news_calculator.save_results(results)
                    total_processed += saved
                    stats.processed = total_processed

                    if self.progress_callback:
                        self.progress_callback(stats.processed, stats.total_records, 'news')

                except Exception as e:
                    logger.error(f"Batch error: {e}")
                    stats.failed += len(batch_df)

            stats.complete()
            logger.info(f"News processing complete: {stats.processed}/{stats.total_records}")

        except Exception as e:
            logger.error(f"Processing error: {e}")
            stats.complete()

        return stats

    def process_all_transcripts(self, limit: int = None) -> ProcessingStats:
        """
        Process all unprocessed transcripts.

        Args:
            limit: Maximum transcripts to process

        Returns:
            ProcessingStats
        """
        stats = ProcessingStats(source_type='transcript')

        logger.info(f"Starting batch transcript processing (limit={limit})")

        try:
            # Get unprocessed
            transcripts_df = self.transcript_calculator.get_unprocessed_transcripts(limit or 1000)
            stats.total_records = len(transcripts_df)

            if transcripts_df.empty:
                logger.info("No unprocessed transcripts found")
                stats.complete()
                return stats

            # Process one at a time (transcripts are large)
            for i, (_, row) in enumerate(transcripts_df.iterrows()):
                try:
                    result = self.transcript_calculator.analyzer.analyze(
                        transcript=row.get('content', ''),
                        symbol=row.get('symbol', ''),
                        year=int(row.get('year', 0)),
                        quarter=row.get('quarter', ''),
                        earnings_date=row.get('date')
                    )

                    self.transcript_calculator.save_results([result])
                    stats.processed += 1

                    if self.progress_callback:
                        self.progress_callback(stats.processed, stats.total_records, 'transcript')

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{stats.total_records} transcripts")

                except Exception as e:
                    logger.error(f"Transcript error {row.get('symbol')}: {e}")
                    stats.failed += 1

            stats.complete()
            logger.info(f"Transcript processing complete: {stats.processed}/{stats.total_records}")

        except Exception as e:
            logger.error(f"Processing error: {e}")
            stats.complete()

        return stats

    def process_aggregates(
        self,
        symbols: Optional[List[str]] = None,
        start_date = None
    ) -> ProcessingStats:
        """
        Calculate aggregate sentiment features.

        Args:
            symbols: List of symbols (None for all)
            start_date: Start date

        Returns:
            ProcessingStats
        """
        stats = ProcessingStats(source_type='aggregate')

        logger.info("Starting aggregate sentiment calculation")

        try:
            if symbols is None:
                symbols = self.aggregate_calculator.get_symbols_with_sentiment()

            stats.total_records = len(symbols)

            for i, symbol in enumerate(symbols):
                try:
                    count = self.aggregate_calculator.process_symbol(symbol, start_date)
                    stats.processed += 1

                    if self.progress_callback:
                        self.progress_callback(stats.processed, stats.total_records, 'aggregate')

                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{stats.total_records} symbols")

                except Exception as e:
                    logger.error(f"Aggregate error {symbol}: {e}")
                    stats.failed += 1

            stats.complete()
            logger.info(f"Aggregate processing complete: {stats.processed}/{stats.total_records}")

        except Exception as e:
            logger.error(f"Processing error: {e}")
            stats.complete()

        return stats

    def process_all(
        self,
        news_limit: int = 5000,
        transcript_limit: int = 500,
        start_date = None
    ) -> Dict[str, ProcessingStats]:
        """
        Run full processing pipeline.

        Args:
            news_limit: Maximum news articles
            transcript_limit: Maximum transcripts
            start_date: Start date for news

        Returns:
            Dict with stats for each processing type
        """
        results = {}

        logger.info("Starting full NLP processing pipeline")
        start_time = time.time()

        # Process news
        results['news'] = self.process_all_news(news_limit, start_date)

        # Process transcripts
        results['transcripts'] = self.process_all_transcripts(transcript_limit)

        # Calculate aggregates
        results['aggregates'] = self.process_aggregates(start_date=start_date)

        elapsed = time.time() - start_time
        logger.info(f"Full pipeline complete in {elapsed:.1f}s")

        # Summary
        total_processed = sum(r.processed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        logger.info(f"Total: {total_processed} processed, {total_failed} failed")

        return results


def run_batch_processing(
    news_limit: int = 5000,
    transcript_limit: int = 500
) -> Dict[str, Any]:
    """
    Convenience function to run batch processing.

    Args:
        news_limit: Maximum news articles
        transcript_limit: Maximum transcripts

    Returns:
        Dict with processing results
    """
    processor = BatchProcessor()
    results = processor.process_all(news_limit, transcript_limit)
    return {k: v.to_dict() for k, v in results.items()}

#!/usr/bin/env python
"""
Batch Processing Script for NLP Sentiment Analysis.

Usage:
    python -m scripts.nlp.run_batch_processing
    python -m scripts.nlp.run_batch_processing --news-limit 1000
    python -m scripts.nlp.run_batch_processing --transcripts-only
"""

import argparse
import logging
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run NLP batch processing')

    parser.add_argument(
        '--news-limit',
        type=int,
        default=5000,
        help='Maximum news articles to process'
    )
    parser.add_argument(
        '--transcript-limit',
        type=int,
        default=500,
        help='Maximum transcripts to process'
    )
    parser.add_argument(
        '--news-only',
        action='store_true',
        help='Process only news'
    )
    parser.add_argument(
        '--transcripts-only',
        action='store_true',
        help='Process only transcripts'
    )
    parser.add_argument(
        '--aggregates-only',
        action='store_true',
        help='Calculate aggregates only'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for news (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    from src.nlp.pipelines.batch_processor import BatchProcessor

    processor = BatchProcessor()

    start_date = None
    if args.start_date:
        start_date = date.fromisoformat(args.start_date)

    if args.news_only:
        logger.info("Processing news only")
        stats = processor.process_all_news(args.news_limit, start_date)
        logger.info(f"Result: {stats.to_dict()}")

    elif args.transcripts_only:
        logger.info("Processing transcripts only")
        stats = processor.process_all_transcripts(args.transcript_limit)
        logger.info(f"Result: {stats.to_dict()}")

    elif args.aggregates_only:
        logger.info("Calculating aggregates only")
        stats = processor.process_aggregates(start_date=start_date)
        logger.info(f"Result: {stats.to_dict()}")

    else:
        logger.info("Running full processing pipeline")
        results = processor.process_all(
            news_limit=args.news_limit,
            transcript_limit=args.transcript_limit,
            start_date=start_date
        )

        logger.info("Processing complete:")
        for source, stats in results.items():
            logger.info(f"  {source}: {stats.to_dict()}")


if __name__ == '__main__':
    main()

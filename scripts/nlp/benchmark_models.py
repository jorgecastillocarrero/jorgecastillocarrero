#!/usr/bin/env python
"""
Benchmark script for NLP models.

Usage:
    python -m scripts.nlp.benchmark_models
"""

import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample financial texts for benchmarking
SAMPLE_TEXTS = [
    "Apple Inc. reported record quarterly revenue of $123 billion, beating analyst expectations.",
    "Tesla shares plunged 10% after disappointing delivery numbers and margin concerns.",
    "The Federal Reserve signaled it may pause interest rate hikes amid cooling inflation.",
    "Microsoft announced a $10 billion investment in OpenAI, boosting AI ambitions.",
    "Amazon's AWS revenue growth slowed to 12%, missing Wall Street estimates.",
    "Goldman Sachs reported a 66% drop in profits, cutting 3,200 jobs globally.",
    "Nvidia's shares surged 25% on strong AI chip demand and raised guidance.",
    "Meta Platforms beat earnings expectations, but ad revenue growth remains weak.",
    "JPMorgan Chase posted record annual profit of $48.3 billion despite economic uncertainty.",
    "Netflix added 7.7 million subscribers, exceeding expectations after password sharing crackdown.",
    "Boeing's 737 Max production continues to face supply chain challenges.",
    "Disney+ lost 4 million subscribers as streaming wars intensify.",
    "Alphabet's Google faces antitrust lawsuit over search monopoly practices.",
    "Pfizer's COVID vaccine revenue dropped 80% as pandemic demand wanes.",
    "Exxon Mobil reported $55.7 billion annual profit, highest in company history.",
]


def benchmark_finbert():
    """Benchmark FinBERT model."""
    from src.nlp.models.finbert import FinBERTModel

    logger.info("Benchmarking FinBERT...")

    model = FinBERTModel()
    model.load()

    # Warm up
    _ = model.predict(SAMPLE_TEXTS[0])

    # Single prediction benchmark
    start = time.time()
    for text in SAMPLE_TEXTS:
        _ = model.predict(text)
    single_time = time.time() - start

    # Batch prediction benchmark
    start = time.time()
    _ = model.predict_batch(SAMPLE_TEXTS)
    batch_time = time.time() - start

    logger.info(f"FinBERT Results:")
    logger.info(f"  Single predictions: {len(SAMPLE_TEXTS)} texts in {single_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/single_time:.1f} texts/s)")
    logger.info(f"  Batch prediction: {len(SAMPLE_TEXTS)} texts in {batch_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/batch_time:.1f} texts/s)")

    model.unload()
    return single_time, batch_time


def benchmark_roberta():
    """Benchmark RoBERTa model."""
    from src.nlp.models.roberta import RoBERTaModel

    logger.info("Benchmarking RoBERTa...")

    model = RoBERTaModel()
    model.load()

    # Warm up
    _ = model.predict(SAMPLE_TEXTS[0])

    # Single prediction benchmark
    start = time.time()
    for text in SAMPLE_TEXTS:
        _ = model.predict(text)
    single_time = time.time() - start

    # Batch prediction benchmark
    start = time.time()
    _ = model.predict_batch(SAMPLE_TEXTS)
    batch_time = time.time() - start

    logger.info(f"RoBERTa Results:")
    logger.info(f"  Single predictions: {len(SAMPLE_TEXTS)} texts in {single_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/single_time:.1f} texts/s)")
    logger.info(f"  Batch prediction: {len(SAMPLE_TEXTS)} texts in {batch_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/batch_time:.1f} texts/s)")

    model.unload()
    return single_time, batch_time


def benchmark_ensemble():
    """Benchmark Ensemble model."""
    from src.nlp.models.ensemble import EnsembleSentimentModel

    logger.info("Benchmarking Ensemble...")

    model = EnsembleSentimentModel()
    model.load()

    # Warm up
    _ = model.predict(SAMPLE_TEXTS[0])

    # Single prediction benchmark
    start = time.time()
    for text in SAMPLE_TEXTS:
        _ = model.predict(text)
    single_time = time.time() - start

    # Batch prediction benchmark
    start = time.time()
    _ = model.predict_batch(SAMPLE_TEXTS)
    batch_time = time.time() - start

    logger.info(f"Ensemble Results:")
    logger.info(f"  Single predictions: {len(SAMPLE_TEXTS)} texts in {single_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/single_time:.1f} texts/s)")
    logger.info(f"  Batch prediction: {len(SAMPLE_TEXTS)} texts in {batch_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/batch_time:.1f} texts/s)")

    # Show sample predictions
    logger.info("\nSample predictions:")
    for text in SAMPLE_TEXTS[:3]:
        result = model.predict(text)
        logger.info(f"  Text: {text[:50]}...")
        logger.info(f"  Result: {result.label} (score={result.score:.2f}, conf={result.confidence:.2f})")

    model.unload()
    return single_time, batch_time


def benchmark_sentiment_service():
    """Benchmark SentimentService (singleton)."""
    from src.nlp.services.sentiment_service import get_sentiment_service, reset_sentiment_service

    logger.info("Benchmarking SentimentService...")

    reset_sentiment_service()
    service = get_sentiment_service()

    # First call loads models
    start = time.time()
    _ = service.analyze(SAMPLE_TEXTS[0])
    init_time = time.time() - start
    logger.info(f"  Initialization time: {init_time:.2f}s")

    # Batch benchmark
    start = time.time()
    _ = service.analyze_batch(SAMPLE_TEXTS)
    batch_time = time.time() - start

    logger.info(f"  Batch analysis: {len(SAMPLE_TEXTS)} texts in {batch_time:.2f}s "
                f"({len(SAMPLE_TEXTS)/batch_time:.1f} texts/s)")

    # Status
    logger.info(f"  Service status: {service.get_status()}")

    return batch_time


def main():
    import torch

    logger.info("=" * 60)
    logger.info("NLP Models Benchmark")
    logger.info("=" * 60)
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Sample texts: {len(SAMPLE_TEXTS)}")
    logger.info("=" * 60)

    # Run benchmarks
    logger.info("\n")
    finbert_times = benchmark_finbert()

    logger.info("\n")
    roberta_times = benchmark_roberta()

    logger.info("\n")
    ensemble_times = benchmark_ensemble()

    logger.info("\n")
    service_time = benchmark_sentiment_service()

    # Summary
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"FinBERT batch: {len(SAMPLE_TEXTS)/finbert_times[1]:.1f} texts/s")
    logger.info(f"RoBERTa batch: {len(SAMPLE_TEXTS)/roberta_times[1]:.1f} texts/s")
    logger.info(f"Ensemble batch: {len(SAMPLE_TEXTS)/ensemble_times[1]:.1f} texts/s")
    logger.info(f"Service batch: {len(SAMPLE_TEXTS)/service_time:.1f} texts/s")


if __name__ == '__main__':
    main()

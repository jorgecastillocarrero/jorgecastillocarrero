"""
Integration tests for NLP module.

These tests require models to be loaded (slower).
Run with: pytest tests/nlp/test_integration.py -v --slow
"""

import pytest
from unittest.mock import patch, MagicMock


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture
def sample_news_texts():
    return [
        "Apple Inc. reported record quarterly revenue, exceeding analyst expectations.",
        "Tesla shares dropped 10% after disappointing delivery numbers.",
        "The Federal Reserve maintained interest rates unchanged.",
    ]


@pytest.fixture
def sample_transcript():
    return """
    [CEO]: Thank you for joining us today. We're pleased to report strong results
    this quarter. Revenue grew 15% year over year, driven by our cloud services.

    [CFO]: Our operating margin improved to 28%, reflecting our cost discipline.
    We're raising our full-year guidance to $50 billion.

    [Operator]: We'll now take questions from analysts.

    [Analyst]: Can you comment on the competitive environment?

    [CEO]: Competition remains intense, but we believe our differentiated products
    give us an advantage. We're investing heavily in AI capabilities.
    """


class TestNewsAnalyzerIntegration:
    """Integration tests for news analyzer."""

    @pytest.mark.skipif(
        True,  # Skip by default - models take time to load
        reason="Requires model loading"
    )
    def test_analyze_single_news(self, sample_news_texts):
        from src.nlp.analyzers.news_analyzer import NewsAnalyzer

        analyzer = NewsAnalyzer()
        result = analyzer.analyze(
            title=sample_news_texts[0],
            content="Apple's quarterly results exceeded expectations with strong iPhone sales.",
            news_id=1
        )

        assert result.ensemble_label in ['positive', 'negative', 'neutral']
        assert -1.0 <= result.ensemble_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0


class TestTranscriptAnalyzerIntegration:
    """Integration tests for transcript analyzer."""

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires model loading"
    )
    def test_analyze_transcript(self, sample_transcript):
        from src.nlp.analyzers.transcript_analyzer import TranscriptAnalyzer

        analyzer = TranscriptAnalyzer()
        result = analyzer.analyze(
            transcript=sample_transcript,
            symbol="TEST",
            year=2024,
            quarter="Q4"
        )

        assert result.overall_label in ['positive', 'negative', 'neutral']
        assert result.num_segments > 0


class TestCacheManager:
    """Tests for cache manager."""

    def test_memory_cache(self):
        from src.nlp.storage.cache import CacheManager

        cache = CacheManager(prefix="test")

        # Set and get
        cache.set("key1", {"value": 123})
        result = cache.get("key1")

        assert result["value"] == 123

    def test_cache_expiry(self):
        from src.nlp.storage.cache import CacheManager
        import time

        cache = CacheManager(prefix="test")

        # Set with short TTL
        cache.set("short_key", "value", ttl_seconds=1)

        # Should exist
        assert cache.get("short_key") == "value"

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert cache.get("short_key") is None

    def test_sentiment_cache(self):
        from src.nlp.storage.cache import CacheManager

        cache = CacheManager(prefix="test")

        # Cache sentiment result
        cache.set_sentiment("test text", {"score": 0.5, "label": "positive"})

        # Retrieve
        result = cache.get_sentiment("test text")

        assert result["score"] == 0.5
        assert result["label"] == "positive"


class TestVectorStore:
    """Tests for vector store."""

    def test_chromadb_init(self):
        """Test ChromaDB initialization (if available)."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        from src.nlp.storage.vector_store import VectorStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.nlp.config.get_nlp_settings') as mock_settings:
                mock_settings.return_value.chromadb_path = tmpdir
                mock_settings.return_value.vector_store_type = 'chromadb'

                store = VectorStore(store_type='chromadb', collection_name='test')
                assert store._ensure_initialized()


class TestProcessingPipeline:
    """Tests for processing pipeline."""

    def test_batch_processor_init(self):
        from src.nlp.pipelines.batch_processor import BatchProcessor

        processor = BatchProcessor(max_workers=2, batch_size=16)

        assert processor.max_workers == 2
        assert processor.batch_size == 16

    def test_incremental_processor_init(self):
        from src.nlp.pipelines.incremental_processor import IncrementalProcessor

        processor = IncrementalProcessor()

        assert processor._storage is None  # Lazy loaded


class TestConfig:
    """Tests for NLP configuration."""

    def test_config_defaults(self):
        from src.nlp.config import NLPSettings

        settings = NLPSettings()

        assert settings.batch_size == 32
        assert settings.finbert_weight == 0.6
        assert settings.roberta_weight == 0.4
        assert settings.finbert_weight + settings.roberta_weight == 1.0

    def test_effective_device(self):
        from src.nlp.config import NLPSettings

        settings = NLPSettings(device="cpu")

        assert settings.effective_device == "cpu"

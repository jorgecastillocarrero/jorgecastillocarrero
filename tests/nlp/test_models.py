"""
Tests for NLP sentiment models.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        from src.nlp.models.base import SentimentResult

        result = SentimentResult(
            text="Test text",
            label="positive",
            score=0.8,
            confidence=0.9,
            model_name="test_model",
            probabilities={"positive": 0.9, "negative": 0.05, "neutral": 0.05}
        )

        assert result.text == "Test text"
        assert result.label == "positive"
        assert result.is_positive is True
        assert result.is_negative is False
        assert result.is_neutral is False

    def test_sentiment_result_to_dict(self):
        from src.nlp.models.base import SentimentResult

        result = SentimentResult(
            text="A" * 1000,  # Long text
            label="negative",
            score=-0.5,
            confidence=0.7,
            model_name="test"
        )

        d = result.to_dict()

        assert len(d['text']) <= 500  # Text is truncated
        assert d['label'] == "negative"
        assert d['score'] == -0.5


class TestTextCleaner:
    """Tests for text cleaning."""

    def test_clean_removes_urls(self):
        from src.nlp.processors.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Check out https://example.com for more info."
        cleaned = cleaner.clean(text)

        assert "https://" not in cleaned
        assert "example.com" not in cleaned

    def test_clean_removes_html(self):
        from src.nlp.processors.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "<p>This is <b>bold</b> text.</p>"
        cleaned = cleaner.clean(text)

        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "bold" in cleaned

    def test_clean_normalizes_whitespace(self):
        from src.nlp.processors.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Multiple   spaces\n\nand\nnewlines"
        cleaned = cleaner.clean(text)

        assert "  " not in cleaned


class TestChunker:
    """Tests for text chunking."""

    def test_short_text_single_chunk(self):
        from src.nlp.processors.chunker import TextChunker

        chunker = TextChunker()
        text = "This is a short text that fits in one chunk."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        from src.nlp.processors.chunker import TextChunker, ChunkingConfig

        config = ChunkingConfig(chunk_size=10, chunk_overlap=2)
        chunker = TextChunker(config)

        # Create text with more than 10 words
        text = " ".join(["word"] * 50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1

    def test_chunk_text_function(self):
        from src.nlp.processors.chunker import chunk_text

        text = " ".join(["sentence"] * 100)
        chunks = chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)


class TestEntityExtractor:
    """Tests for entity extraction."""

    def test_extract_cashtags(self):
        from src.nlp.processors.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        text = "$AAPL and $MSFT are tech stocks."
        result = extractor.extract(text)

        assert "AAPL" in result.tickers
        assert "MSFT" in result.tickers

    def test_extract_money(self):
        from src.nlp.processors.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        text = "Revenue was $1.5 billion and margins improved by 3.5%."
        result = extractor.extract(text)

        assert len(result.money_amounts) > 0

    def test_extract_tickers_function(self):
        from src.nlp.processors.entity_extractor import extract_tickers

        text = "Apple ($AAPL) reported earnings."
        tickers = extract_tickers(text)

        assert "AAPL" in tickers


class TestSentimentService:
    """Tests for sentiment service."""

    def test_fallback_analyze(self):
        from src.nlp.services.sentiment_service import SentimentService

        service = SentimentService()
        # Test fallback without loading models
        result = service._fallback_analyze("This is great news for investors!")

        assert result.label in ['positive', 'negative', 'neutral']
        assert result.model_name == 'fallback'

    def test_fallback_empty_text(self):
        from src.nlp.services.sentiment_service import SentimentService

        service = SentimentService()
        result = service._fallback_analyze("")

        assert result.label == 'neutral'
        assert result.confidence == 0.0


class TestNewsAnalysisResult:
    """Tests for news analysis result."""

    def test_to_dict(self):
        from src.nlp.analyzers.news_analyzer import NewsAnalysisResult
        from datetime import datetime

        result = NewsAnalysisResult(
            news_id=123,
            title="Test Title",
            content="Test content",
            ensemble_score=0.5,
            ensemble_label="positive",
            confidence=0.8,
            tickers=["AAPL"],
            processed_at=datetime.now()
        )

        d = result.to_dict()

        assert d['news_id'] == 123
        assert d['ensemble_score'] == 0.5
        assert d['tickers'] == ["AAPL"]

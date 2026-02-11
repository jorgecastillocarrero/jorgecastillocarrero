"""
Sentiment Service Singleton.
Central service for all sentiment analysis operations.
"""

import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

from ..models.base import SentimentResult
from ..models.ensemble import EnsembleSentimentModel
from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class SentimentService:
    """
    Central sentiment analysis service.

    Provides a unified interface for sentiment analysis with:
    - Lazy model loading
    - Caching support
    - Batch processing
    - Fallback handling
    """

    def __init__(self):
        """Initialize sentiment service."""
        self.settings = get_nlp_settings()
        self._model: Optional[EnsembleSentimentModel] = None
        self._is_initialized = False
        self._fallback_mode = False

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded."""
        if self._is_initialized:
            return not self._fallback_mode

        logger.info("Initializing sentiment service...")

        try:
            self._model = EnsembleSentimentModel(
                device=self.settings.effective_device
            )

            if self._model.load():
                self._is_initialized = True
                logger.info(f"Sentiment service initialized on {self.settings.effective_device}")
                return True
            else:
                logger.warning("Failed to load models, falling back to simple analysis")
                self._fallback_mode = True
                self._is_initialized = True
                return False

        except Exception as e:
            logger.error(f"Error initializing sentiment service: {e}")
            self._fallback_mode = True
            self._is_initialized = True
            return False

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text

        Returns:
            SentimentResult
        """
        if not self._ensure_loaded():
            return self._fallback_analyze(text)

        try:
            return self._model.predict(text)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._fallback_analyze(text)

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        if not self._ensure_loaded():
            return [self._fallback_analyze(text) for text in texts]

        try:
            return self._model.predict_batch(texts, batch_size)
        except Exception as e:
            logger.error(f"Batch analysis error: {e}")
            return [self._fallback_analyze(text) for text in texts]

    def _fallback_analyze(self, text: str) -> SentimentResult:
        """
        Fallback sentiment analysis using keywords.

        Used when transformer models are not available.
        """
        from ..config import FINANCIAL_KEYWORDS

        if not text:
            return SentimentResult(
                text=text,
                label='neutral',
                score=0.0,
                confidence=0.0,
                model_name='fallback',
                probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            )

        text_lower = text.lower()

        # Count keywords
        positive_count = sum(
            text_lower.count(word) for word in FINANCIAL_KEYWORDS['positive']
        )
        negative_count = sum(
            text_lower.count(word) for word in FINANCIAL_KEYWORDS['negative']
        )

        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            label = 'neutral'
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / total
            confidence = min(0.8, total / 10)

            if score > 0.2:
                label = 'positive'
            elif score < -0.2:
                label = 'negative'
            else:
                label = 'neutral'

        return SentimentResult(
            text=text,
            label=label,
            score=score,
            confidence=confidence,
            model_name='fallback',
            probabilities={
                'positive': max(0, (score + 1) / 2),
                'negative': max(0, (1 - score) / 2),
                'neutral': 1 - abs(score)
            },
            metadata={'method': 'keyword_based'}
        )

    def get_individual_predictions(
        self,
        text: str
    ) -> Dict[str, SentimentResult]:
        """
        Get predictions from each model separately.

        Args:
            text: Input text

        Returns:
            Dict mapping model name to result
        """
        if not self._ensure_loaded():
            return {'fallback': self._fallback_analyze(text)}

        return self._model.get_individual_predictions(text)

    def is_available(self) -> bool:
        """Check if service is available with transformer models."""
        self._ensure_loaded()
        return not self._fallback_mode

    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            'initialized': self._is_initialized,
            'fallback_mode': self._fallback_mode,
            'device': self.settings.effective_device,
            'gpu_available': self.settings.is_gpu_available,
            'models': {
                'finbert': self.settings.finbert_model,
                'roberta': self.settings.roberta_model,
            },
            'weights': {
                'finbert': self.settings.finbert_weight,
                'roberta': self.settings.roberta_weight,
            }
        }

    def unload(self) -> None:
        """Unload models to free memory."""
        if self._model:
            self._model.unload()
            self._model = None

        self._is_initialized = False
        self._fallback_mode = False
        logger.info("Sentiment service unloaded")


# Singleton instance
_sentiment_service: Optional[SentimentService] = None


def get_sentiment_service() -> SentimentService:
    """
    Get the singleton sentiment service instance.

    Returns:
        SentimentService instance
    """
    global _sentiment_service

    if _sentiment_service is None:
        _sentiment_service = SentimentService()

    return _sentiment_service


def reset_sentiment_service() -> None:
    """Reset the singleton (useful for testing)."""
    global _sentiment_service

    if _sentiment_service:
        _sentiment_service.unload()

    _sentiment_service = None

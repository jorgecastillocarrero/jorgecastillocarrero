"""
Ensemble Sentiment Model.
Combines multiple models for robust sentiment analysis.
"""

import logging
from typing import List, Optional, Dict
import numpy as np

from .base import BaseSentimentModel, SentimentResult
from .finbert import FinBERTModel
from .roberta import RoBERTaModel
from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class EnsembleSentimentModel(BaseSentimentModel):
    """
    Ensemble model combining FinBERT and RoBERTa.

    Weights can be configured to favor financial-specific (FinBERT)
    or general (RoBERTa) sentiment depending on use case.
    """

    def __init__(
        self,
        finbert_weight: float = None,
        roberta_weight: float = None,
        device: str = "auto"
    ):
        """
        Initialize ensemble model.

        Args:
            finbert_weight: Weight for FinBERT (0-1)
            roberta_weight: Weight for RoBERTa (0-1)
            device: Device to use
        """
        super().__init__("ensemble", device)

        settings = get_nlp_settings()

        self.finbert_weight = finbert_weight or settings.finbert_weight
        self.roberta_weight = roberta_weight or settings.roberta_weight

        # Normalize weights
        total = self.finbert_weight + self.roberta_weight
        self.finbert_weight /= total
        self.roberta_weight /= total

        # Initialize component models
        self._finbert: Optional[FinBERTModel] = None
        self._roberta: Optional[RoBERTaModel] = None

    def load(self) -> bool:
        """Load all component models."""
        try:
            logger.info("Loading ensemble models...")

            # Load FinBERT
            self._finbert = FinBERTModel(device=self._device)
            if not self._finbert.load():
                logger.warning("FinBERT failed to load, using RoBERTa only")
                self.finbert_weight = 0.0
                self.roberta_weight = 1.0

            # Load RoBERTa
            self._roberta = RoBERTaModel(device=self._device)
            if not self._roberta.load():
                logger.warning("RoBERTa failed to load, using FinBERT only")
                self.roberta_weight = 0.0
                self.finbert_weight = 1.0

            # Check at least one model loaded
            if not self._finbert.is_loaded and not self._roberta.is_loaded:
                logger.error("No models loaded successfully")
                return False

            self._is_loaded = True
            logger.info(
                f"Ensemble loaded: FinBERT={self.finbert_weight:.1%}, "
                f"RoBERTa={self.roberta_weight:.1%}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False

    def predict(self, text: str) -> SentimentResult:
        """
        Predict sentiment using ensemble of models.

        Args:
            text: Input text

        Returns:
            SentimentResult with weighted ensemble prediction
        """
        self._ensure_loaded()

        # Handle empty text
        if not text or len(text.strip()) < 3:
            return SentimentResult(
                text=text,
                label='neutral',
                score=0.0,
                confidence=0.0,
                model_name='ensemble',
                probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            )

        # Get predictions from each model
        results = {}

        if self._finbert and self._finbert.is_loaded and self.finbert_weight > 0:
            results['finbert'] = self._finbert.predict(text)

        if self._roberta and self._roberta.is_loaded and self.roberta_weight > 0:
            results['roberta'] = self._roberta.predict(text)

        if not results:
            return SentimentResult(
                text=text,
                label='neutral',
                score=0.0,
                confidence=0.0,
                model_name='ensemble',
                probabilities={}
            )

        # Combine results
        return self._combine_results(text, results)

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Predict sentiment for multiple texts using ensemble.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects
        """
        self._ensure_loaded()

        # Get batch predictions from each model
        finbert_results = []
        roberta_results = []

        if self._finbert and self._finbert.is_loaded and self.finbert_weight > 0:
            finbert_results = self._finbert.predict_batch(texts, batch_size)

        if self._roberta and self._roberta.is_loaded and self.roberta_weight > 0:
            roberta_results = self._roberta.predict_batch(texts, batch_size)

        # Combine results for each text
        ensemble_results = []
        for i, text in enumerate(texts):
            results = {}

            if finbert_results:
                results['finbert'] = finbert_results[i]
            if roberta_results:
                results['roberta'] = roberta_results[i]

            if results:
                ensemble_results.append(self._combine_results(text, results))
            else:
                ensemble_results.append(SentimentResult(
                    text=text,
                    label='neutral',
                    score=0.0,
                    confidence=0.0,
                    model_name='ensemble',
                    probabilities={}
                ))

        return ensemble_results

    def _combine_results(
        self,
        text: str,
        results: Dict[str, SentimentResult]
    ) -> SentimentResult:
        """
        Combine results from multiple models.

        Args:
            text: Original text
            results: Dict mapping model name to SentimentResult

        Returns:
            Combined SentimentResult
        """
        # Weighted average of scores
        weighted_score = 0.0
        weighted_confidence = 0.0
        combined_probs = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        weights = {
            'finbert': self.finbert_weight,
            'roberta': self.roberta_weight
        }

        total_weight = 0.0
        for model_name, result in results.items():
            weight = weights.get(model_name, 0.0)
            if weight > 0:
                weighted_score += result.score * weight
                weighted_confidence += result.confidence * weight

                for label in combined_probs:
                    combined_probs[label] += result.probabilities.get(label, 0.33) * weight

                total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_score /= total_weight
            weighted_confidence /= total_weight
            for label in combined_probs:
                combined_probs[label] /= total_weight

        # Determine final label
        if weighted_score > 0.1:
            label = 'positive'
        elif weighted_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # Store individual model results in metadata
        metadata = {
            'models': {
                name: {
                    'label': r.label,
                    'score': r.score,
                    'confidence': r.confidence
                }
                for name, r in results.items()
            },
            'weights': weights
        }

        return SentimentResult(
            text=text,
            label=label,
            score=weighted_score,
            confidence=weighted_confidence,
            model_name='ensemble',
            probabilities=combined_probs,
            metadata=metadata
        )

    def unload(self) -> None:
        """Unload all models."""
        if self._finbert:
            self._finbert.unload()
        if self._roberta:
            self._roberta.unload()

        super().unload()
        logger.info("Ensemble models unloaded")

    def get_individual_predictions(
        self,
        text: str
    ) -> Dict[str, SentimentResult]:
        """
        Get predictions from each model separately.

        Useful for debugging and analysis.

        Args:
            text: Input text

        Returns:
            Dict mapping model name to SentimentResult
        """
        self._ensure_loaded()

        results = {}

        if self._finbert and self._finbert.is_loaded:
            results['finbert'] = self._finbert.predict(text)

        if self._roberta and self._roberta.is_loaded:
            results['roberta'] = self._roberta.predict(text)

        return results

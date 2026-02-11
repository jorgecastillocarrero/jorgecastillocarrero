"""
Base Sentiment Model Interface.
Abstract base class for all sentiment analysis models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    text: str
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # -1.0 to 1.0 (normalized)
    confidence: float  # 0.0 to 1.0
    model_name: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_positive(self) -> bool:
        return self.label == 'positive'

    @property
    def is_negative(self) -> bool:
        return self.label == 'negative'

    @property
    def is_neutral(self) -> bool:
        return self.label == 'neutral'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'text': self.text[:500],  # Truncate for storage
            'label': self.label,
            'score': self.score,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'probabilities': self.probabilities,
            'metadata': self.metadata,
        }


class BaseSentimentModel(ABC):
    """Abstract base class for sentiment models."""

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize base sentiment model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    def device(self) -> str:
        """Get the actual device being used."""
        if self._device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model and tokenizer.

        Returns:
            True if loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def predict(self, text: str) -> SentimentResult:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text to analyze

        Returns:
            SentimentResult with prediction
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects
        """
        pass

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        # Force garbage collection
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        import gc
        gc.collect()
        logger.info(f"Unloaded model: {self.model_name}")

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before prediction."""
        if not self._is_loaded:
            if not self.load():
                raise RuntimeError(f"Failed to load model: {self.model_name}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device}, loaded={self._is_loaded})"

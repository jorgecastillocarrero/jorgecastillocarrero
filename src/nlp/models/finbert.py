"""
FinBERT Sentiment Model.
Uses ProsusAI/finbert for financial text sentiment analysis.
"""

import logging
from typing import List, Optional
import numpy as np

from .base import BaseSentimentModel, SentimentResult

logger = logging.getLogger(__name__)


class FinBERTModel(BaseSentimentModel):
    """
    FinBERT model for financial sentiment analysis.

    FinBERT is pre-trained on financial texts and fine-tuned for
    sentiment analysis on financial news and documents.

    Labels: positive, negative, neutral
    """

    # Label mapping for FinBERT output
    LABEL_MAP = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'LABEL_0': 'positive',
        'LABEL_1': 'negative',
        'LABEL_2': 'neutral',
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize FinBERT model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device)
        self.max_length = max_length
        self._pipeline = None

    def load(self) -> bool:
        """Load FinBERT model and tokenizer."""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline
            )
            import torch

            logger.info(f"Loading FinBERT model: {self.model_name}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Move to device
            device_id = 0 if self.device == "cuda" else -1
            self._model.to(self.device)

            # Create pipeline for easier inference
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device_id if self.device == "cuda" else -1,
                max_length=self.max_length,
                truncation=True
            )

            self._is_loaded = True
            logger.info(f"FinBERT loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return False

    def predict(self, text: str) -> SentimentResult:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text

        Returns:
            SentimentResult
        """
        self._ensure_loaded()

        try:
            # Handle empty or very short text
            if not text or len(text.strip()) < 3:
                return SentimentResult(
                    text=text,
                    label='neutral',
                    score=0.0,
                    confidence=0.0,
                    model_name=self.model_name,
                    probabilities={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                )

            # Get prediction
            result = self._pipeline(text[:self.max_length * 4])[0]  # Rough char limit

            # Normalize label
            raw_label = result['label']
            label = self.LABEL_MAP.get(raw_label, 'neutral')
            confidence = result['score']

            # Convert to normalized score (-1 to 1)
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:
                score = 0.0

            # Get all probabilities (requires forward pass)
            probs = self._get_probabilities(text)

            return SentimentResult(
                text=text,
                label=label,
                score=score,
                confidence=confidence,
                model_name=self.model_name,
                probabilities=probs
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return SentimentResult(
                text=text,
                label='neutral',
                score=0.0,
                confidence=0.0,
                model_name=self.model_name,
                probabilities={},
                metadata={'error': str(e)}
            )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects
        """
        self._ensure_loaded()

        results = []

        # Filter and prepare texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) >= 3:
                valid_texts.append(text[:self.max_length * 4])
                valid_indices.append(i)

        # Process in batches
        try:
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_results = self._pipeline(batch)

                for j, result in enumerate(batch_results):
                    raw_label = result['label']
                    label = self.LABEL_MAP.get(raw_label, 'neutral')
                    confidence = result['score']

                    if label == 'positive':
                        score = confidence
                    elif label == 'negative':
                        score = -confidence
                    else:
                        score = 0.0

                    results.append(SentimentResult(
                        text=batch[j],
                        label=label,
                        score=score,
                        confidence=confidence,
                        model_name=self.model_name,
                        probabilities={}
                    ))

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")

        # Insert neutral results for invalid texts
        final_results = [None] * len(texts)
        result_idx = 0
        for i, text in enumerate(texts):
            if i in valid_indices:
                final_results[i] = results[result_idx]
                result_idx += 1
            else:
                final_results[i] = SentimentResult(
                    text=text or "",
                    label='neutral',
                    score=0.0,
                    confidence=0.0,
                    model_name=self.model_name,
                    probabilities={}
                )

        return final_results

    def _get_probabilities(self, text: str) -> dict:
        """Get probability distribution for all classes."""
        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

            # Map to labels (FinBERT order: positive, negative, neutral)
            return {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }

        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
            return {}

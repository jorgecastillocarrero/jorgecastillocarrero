"""
RoBERTa Sentiment Model.
Uses cardiffnlp/twitter-roberta-base-sentiment for general sentiment analysis.
"""

import logging
from typing import List, Optional
import numpy as np

from .base import BaseSentimentModel, SentimentResult

logger = logging.getLogger(__name__)


class RoBERTaModel(BaseSentimentModel):
    """
    RoBERTa model for sentiment analysis.

    Uses Twitter-RoBERTa model fine-tuned on ~124M tweets
    for general sentiment classification.

    Labels: positive, negative, neutral
    """

    # Label mapping for RoBERTa output
    LABEL_MAP = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'LABEL_0': 'negative',  # twitter-roberta order
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize RoBERTa model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use
            max_length: Maximum sequence length
        """
        super().__init__(model_name, device)
        self.max_length = max_length
        self._pipeline = None

    def load(self) -> bool:
        """Load RoBERTa model and tokenizer."""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline
            )
            import torch

            logger.info(f"Loading RoBERTa model: {self.model_name}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Move to device
            device_id = 0 if self.device == "cuda" else -1
            self._model.to(self.device)

            # Create pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device_id if self.device == "cuda" else -1,
                max_length=self.max_length,
                truncation=True,
                top_k=None  # Return all scores
            )

            self._is_loaded = True
            logger.info(f"RoBERTa loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load RoBERTa: {e}")
            return False

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for RoBERTa.
        Handles Twitter-specific tokens.
        """
        # Replace user mentions and URLs (Twitter style)
        import re

        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'http\S+|www\.\S+', 'http', text)

        return text[:self.max_length * 4]

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

            # Preprocess
            processed_text = self._preprocess_text(text)

            # Get prediction with all scores
            results = self._pipeline(processed_text)

            # Results is a list of dicts with label and score
            probs = {}
            best_label = 'neutral'
            best_score = 0.0

            for item in results:
                label = self.LABEL_MAP.get(item['label'], item['label'])
                score = item['score']
                probs[label] = score

                if score > best_score:
                    best_score = score
                    best_label = label

            # Normalize to -1 to 1
            if best_label == 'positive':
                normalized_score = probs.get('positive', 0) - probs.get('negative', 0)
            elif best_label == 'negative':
                normalized_score = probs.get('positive', 0) - probs.get('negative', 0)
            else:
                normalized_score = 0.0

            return SentimentResult(
                text=text,
                label=best_label,
                score=normalized_score,
                confidence=best_score,
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
                valid_texts.append(self._preprocess_text(text))
                valid_indices.append(i)

        # Process in batches
        try:
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]

                # Pipeline returns list of lists when top_k=None
                batch_results = self._pipeline(batch)

                for j, result_list in enumerate(batch_results):
                    probs = {}
                    best_label = 'neutral'
                    best_score = 0.0

                    for item in result_list:
                        label = self.LABEL_MAP.get(item['label'], item['label'])
                        score = item['score']
                        probs[label] = score

                        if score > best_score:
                            best_score = score
                            best_label = label

                    normalized_score = probs.get('positive', 0) - probs.get('negative', 0)

                    results.append(SentimentResult(
                        text=batch[j],
                        label=best_label,
                        score=normalized_score,
                        confidence=best_score,
                        model_name=self.model_name,
                        probabilities=probs
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

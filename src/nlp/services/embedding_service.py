"""
Embedding Service.
Service for text embedding generation and storage.
"""

import logging
from typing import Optional, List, Dict, Any, Union
import numpy as np

from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Text embedding service.

    Provides:
    - Text embedding generation
    - Batch processing
    - Caching support
    - Similarity search helpers
    """

    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_nlp_settings()
        self._model = None
        self._is_initialized = False

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded."""
        if self._is_initialized:
            return True

        logger.info(f"Loading embedding model: {self.settings.embedding_model}")

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.effective_device
            )

            self._is_initialized = True
            logger.info(f"Embedding model loaded on {self.settings.effective_device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Numpy array with embedding or None
        """
        if not self._ensure_loaded():
            return None

        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> Optional[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Numpy array with embeddings (N x dim) or None
        """
        if not texts:
            return np.array([])

        if not self._ensure_loaded():
            return None

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return None

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0 to 1)
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))

    def find_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings in a corpus.

        Args:
            query_embedding: Query embedding
            corpus_embeddings: Corpus of embeddings (N x dim)
            top_k: Number of results to return

        Returns:
            List of dicts with index and score
        """
        # Calculate similarities
        similarities = np.dot(corpus_embeddings, query_embedding)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx])
            })

        return results

    def text_similarity(
        self,
        text1: str,
        text2: str
    ) -> Optional[float]:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0 to 1) or None
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        if emb1 is None or emb2 is None:
            return None

        return self.similarity(emb1, emb2)

    def search_similar_texts(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts in a corpus.

        Args:
            query: Query text
            corpus: List of texts to search
            top_k: Number of results

        Returns:
            List of dicts with text, index, and score
        """
        query_emb = self.embed(query)
        if query_emb is None:
            return []

        corpus_embs = self.embed_batch(corpus)
        if corpus_embs is None:
            return []

        similar = self.find_similar(query_emb, corpus_embs, top_k)

        results = []
        for item in similar:
            idx = item['index']
            results.append({
                'text': corpus[idx],
                'index': idx,
                'score': item['score']
            })

        return results

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.settings.embedding_dim

    def is_available(self) -> bool:
        """Check if service is available."""
        return self._ensure_loaded()

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._is_initialized = False

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Embedding service unloaded")


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the singleton embedding service instance.

    Returns:
        EmbeddingService instance
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service


def reset_embedding_service() -> None:
    """Reset the singleton (useful for testing)."""
    global _embedding_service

    if _embedding_service:
        _embedding_service.unload()

    _embedding_service = None

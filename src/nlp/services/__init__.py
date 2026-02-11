"""
Services Module.
Singleton services for NLP operations.
"""

from .sentiment_service import SentimentService, get_sentiment_service
from .embedding_service import EmbeddingService, get_embedding_service

__all__ = [
    "SentimentService",
    "get_sentiment_service",
    "EmbeddingService",
    "get_embedding_service",
]

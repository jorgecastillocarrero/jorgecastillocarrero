"""
NLP Module Configuration.
Uses pydantic-settings for environment variable handling.
"""

from pathlib import Path
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class NLPSettings(BaseSettings):
    """NLP module settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="NLP_",
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql://fmp:fmp123@localhost:5433/fmp_data",
        description="PostgreSQL connection URL for NLP data"
    )

    # Model Configuration
    finbert_model: str = Field(
        default="ProsusAI/finbert",
        description="FinBERT model for financial sentiment"
    )
    roberta_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="RoBERTa model for general sentiment"
    )

    # Processing Configuration
    batch_size: int = Field(default=32, description="Batch size for inference")
    max_sequence_length: int = Field(default=512, description="Max tokens per sequence")
    chunk_size: int = Field(default=450, description="Chunk size for long texts")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")

    # Ensemble Weights
    finbert_weight: float = Field(default=0.6, description="FinBERT weight in ensemble")
    roberta_weight: float = Field(default=0.4, description="RoBERTa weight in ensemble")

    # Cache Configuration
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    enable_cache: bool = Field(default=False, description="Enable Redis caching")

    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model for text embeddings"
    )
    embedding_dim: int = Field(default=768, description="Embedding dimension")

    # Vector Store Configuration
    vector_store_type: str = Field(
        default="chromadb",
        description="Vector store: chromadb, pgvector, pinecone"
    )
    chromadb_path: str = Field(
        default="data/nlp_chromadb",
        description="Path for ChromaDB persistence"
    )

    # Processing Limits
    max_workers: int = Field(default=4, description="Max parallel workers")
    rate_limit_per_second: float = Field(default=10.0, description="API rate limit")

    # Device Configuration
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    use_fp16: bool = Field(default=True, description="Use FP16 for inference")

    @property
    def effective_device(self) -> str:
        """Get the actual device to use."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


@lru_cache
def get_nlp_settings() -> NLPSettings:
    """Get cached NLP settings instance."""
    return NLPSettings()


# Default ticker patterns for entity extraction
TICKER_PATTERNS = [
    r'\b([A-Z]{1,5})\b',  # Simple tickers like AAPL, MSFT
    r'\$([A-Z]{1,5})\b',  # Cashtag format like $AAPL
]

# Financial keywords for relevance scoring
FINANCIAL_KEYWORDS = {
    'positive': [
        'beat', 'beats', 'exceeds', 'exceeded', 'outperform', 'growth', 'profit',
        'surge', 'soar', 'rally', 'gains', 'bullish', 'upgrade', 'buy', 'strong',
        'record', 'breakthrough', 'innovation', 'expansion', 'dividend'
    ],
    'negative': [
        'miss', 'missed', 'disappoints', 'decline', 'loss', 'losses', 'drop',
        'plunge', 'crash', 'bearish', 'downgrade', 'sell', 'weak', 'warning',
        'recession', 'layoffs', 'bankruptcy', 'default', 'fraud', 'scandal'
    ],
    'neutral': [
        'holds', 'steady', 'unchanged', 'flat', 'mixed', 'stable', 'moderate',
        'inline', 'expected', 'forecast', 'guidance', 'outlook'
    ]
}

# Sections to identify in earnings transcripts
TRANSCRIPT_SECTIONS = [
    'prepared_remarks',
    'q_and_a',
    'guidance',
    'closing_remarks'
]

"""
NLP/Sentiment Analysis Module for Financial Data Project.

This module provides transformer-based sentiment analysis for:
- News articles
- Earnings call transcripts
- Financial filings

Designed for scalability from current ~35 GB to 1 TB.
"""

from .config import get_nlp_settings, NLPSettings

__version__ = "0.1.0"
__all__ = ["get_nlp_settings", "NLPSettings"]

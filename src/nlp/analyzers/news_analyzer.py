"""
News Analyzer for financial news articles.
Combines sentiment analysis with entity extraction.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..models.base import SentimentResult
from ..processors.text_cleaner import NewsTextCleaner
from ..processors.chunker import TextChunker
from ..processors.entity_extractor import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class NewsAnalysisResult:
    """Result of news article analysis."""
    # Original data
    news_id: Optional[int] = None
    title: str = ""
    content: str = ""
    published_date: Optional[datetime] = None
    source: str = ""

    # Sentiment results
    title_sentiment: Optional[SentimentResult] = None
    content_sentiment: Optional[SentimentResult] = None
    overall_sentiment: Optional[SentimentResult] = None

    # Scores (normalized -1 to 1)
    finbert_score: Optional[float] = None
    roberta_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    ensemble_label: str = "neutral"
    confidence: float = 0.0

    # Extracted entities
    tickers: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    money_amounts: List[str] = field(default_factory=list)

    # Metadata
    processed_at: datetime = field(default_factory=datetime.now)
    model_version: str = ""
    num_chunks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'news_id': self.news_id,
            'title': self.title[:500],
            'source': self.source,
            'published_date': self.published_date,
            'finbert_score': self.finbert_score,
            'roberta_score': self.roberta_score,
            'ensemble_score': self.ensemble_score,
            'ensemble_label': self.ensemble_label,
            'confidence': self.confidence,
            'tickers': self.tickers,
            'processed_at': self.processed_at,
            'model_version': self.model_version,
        }


class NewsAnalyzer:
    """
    Analyzer for financial news articles.

    Combines:
    - Text cleaning and preprocessing
    - Sentiment analysis (title + content)
    - Entity extraction (tickers, companies)
    - Relevance scoring
    """

    def __init__(self, sentiment_service=None):
        """
        Initialize news analyzer.

        Args:
            sentiment_service: Optional sentiment service instance
        """
        self._sentiment_service = sentiment_service
        self._cleaner = NewsTextCleaner()
        self._chunker = TextChunker()
        self._extractor = EntityExtractor()

    def _get_sentiment_service(self):
        """Get or create sentiment service."""
        if self._sentiment_service is None:
            from ..services.sentiment_service import get_sentiment_service
            self._sentiment_service = get_sentiment_service()
        return self._sentiment_service

    def analyze(
        self,
        title: str,
        content: str,
        news_id: Optional[int] = None,
        published_date: Optional[datetime] = None,
        source: str = ""
    ) -> NewsAnalysisResult:
        """
        Analyze a single news article.

        Args:
            title: Article title
            content: Article content
            news_id: Database ID
            published_date: Publication date
            source: News source

        Returns:
            NewsAnalysisResult
        """
        result = NewsAnalysisResult(
            news_id=news_id,
            title=title,
            content=content,
            published_date=published_date,
            source=source
        )

        try:
            # Clean texts
            clean_title = self._cleaner.clean(title)
            clean_content = self._cleaner.clean(content)

            # Extract entities from both
            title_entities = self._extractor.extract(title)
            content_entities = self._extractor.extract(content)

            # Combine entities
            all_tickers = list(set(title_entities.tickers + content_entities.tickers))
            all_companies = list(set(title_entities.companies + content_entities.companies))
            all_money = list(set(title_entities.money_amounts + content_entities.money_amounts))

            result.tickers = all_tickers
            result.companies = all_companies
            result.money_amounts = all_money

            # Get sentiment service
            service = self._get_sentiment_service()

            # Analyze title sentiment
            if clean_title:
                result.title_sentiment = service.analyze(clean_title)

            # Analyze content sentiment
            if clean_content:
                # Check if content needs chunking
                chunks = self._chunker.chunk(clean_content)
                result.num_chunks = len(chunks)

                if len(chunks) > 1:
                    # Analyze each chunk and aggregate
                    chunk_sentiments = []
                    for chunk in chunks:
                        chunk_result = service.analyze(chunk.text)
                        chunk_sentiments.append(chunk_result)

                    # Weighted average by chunk length
                    result.content_sentiment = self._aggregate_chunk_sentiments(
                        chunk_sentiments, chunks
                    )
                else:
                    result.content_sentiment = service.analyze(clean_content)

            # Calculate overall sentiment
            result.overall_sentiment = self._calculate_overall_sentiment(
                result.title_sentiment,
                result.content_sentiment
            )

            # Extract scores
            if result.overall_sentiment:
                result.ensemble_score = result.overall_sentiment.score
                result.ensemble_label = result.overall_sentiment.label
                result.confidence = result.overall_sentiment.confidence

                # Get individual model scores from metadata
                if 'models' in result.overall_sentiment.metadata:
                    models = result.overall_sentiment.metadata['models']
                    if 'finbert' in models:
                        result.finbert_score = models['finbert']['score']
                    if 'roberta' in models:
                        result.roberta_score = models['roberta']['score']

            result.model_version = "ensemble_v1"

        except Exception as e:
            logger.error(f"Error analyzing news {news_id}: {e}")
            result.ensemble_label = 'neutral'
            result.ensemble_score = 0.0

        return result

    def analyze_batch(
        self,
        articles: List[Dict],
        batch_size: int = 32
    ) -> List[NewsAnalysisResult]:
        """
        Analyze multiple news articles.

        Args:
            articles: List of article dicts with title, content, etc.
            batch_size: Batch size for sentiment analysis

        Returns:
            List of NewsAnalysisResult
        """
        results = []

        # Prepare texts for batch processing
        all_texts = []
        text_map = []  # (article_idx, 'title'/'content', chunk_idx)

        for i, article in enumerate(articles):
            title = self._cleaner.clean(article.get('title', ''))
            content = self._cleaner.clean(article.get('content', ''))

            if title:
                all_texts.append(title)
                text_map.append((i, 'title', 0))

            if content:
                chunks = self._chunker.chunk(content)
                for j, chunk in enumerate(chunks):
                    all_texts.append(chunk.text)
                    text_map.append((i, 'content', j))

        # Batch sentiment analysis
        service = self._get_sentiment_service()
        all_sentiments = service.analyze_batch(all_texts, batch_size)

        # Map results back to articles
        article_sentiments = {i: {'title': None, 'content': []}
                            for i in range(len(articles))}

        for (article_idx, text_type, chunk_idx), sentiment in zip(text_map, all_sentiments):
            if text_type == 'title':
                article_sentiments[article_idx]['title'] = sentiment
            else:
                article_sentiments[article_idx]['content'].append(sentiment)

        # Build final results
        for i, article in enumerate(articles):
            result = NewsAnalysisResult(
                news_id=article.get('id'),
                title=article.get('title', ''),
                content=article.get('content', ''),
                published_date=article.get('published_date'),
                source=article.get('source', '')
            )

            # Extract entities
            entities = self._extractor.extract(
                f"{article.get('title', '')} {article.get('content', '')}"
            )
            result.tickers = entities.tickers
            result.companies = entities.companies

            # Set sentiments
            result.title_sentiment = article_sentiments[i]['title']

            content_sents = article_sentiments[i]['content']
            if content_sents:
                if len(content_sents) == 1:
                    result.content_sentiment = content_sents[0]
                else:
                    result.content_sentiment = self._aggregate_sentiments(content_sents)

            # Calculate overall
            result.overall_sentiment = self._calculate_overall_sentiment(
                result.title_sentiment,
                result.content_sentiment
            )

            if result.overall_sentiment:
                result.ensemble_score = result.overall_sentiment.score
                result.ensemble_label = result.overall_sentiment.label
                result.confidence = result.overall_sentiment.confidence

            result.model_version = "ensemble_v1"
            results.append(result)

        return results

    def _aggregate_chunk_sentiments(
        self,
        sentiments: List[SentimentResult],
        chunks
    ) -> SentimentResult:
        """Aggregate sentiments from multiple chunks."""
        if not sentiments:
            return None

        # Weight by chunk length
        total_weight = sum(len(c.text) for c in chunks)
        weighted_score = 0.0
        weighted_confidence = 0.0

        for sent, chunk in zip(sentiments, chunks):
            weight = len(chunk.text) / total_weight
            weighted_score += sent.score * weight
            weighted_confidence += sent.confidence * weight

        # Determine label
        if weighted_score > 0.1:
            label = 'positive'
        elif weighted_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return SentimentResult(
            text=f"[Aggregated {len(chunks)} chunks]",
            label=label,
            score=weighted_score,
            confidence=weighted_confidence,
            model_name='aggregate',
            probabilities={},
            metadata={'num_chunks': len(chunks)}
        )

    def _aggregate_sentiments(self, sentiments: List[SentimentResult]) -> SentimentResult:
        """Simple average of sentiments."""
        if not sentiments:
            return None

        avg_score = sum(s.score for s in sentiments) / len(sentiments)
        avg_conf = sum(s.confidence for s in sentiments) / len(sentiments)

        if avg_score > 0.1:
            label = 'positive'
        elif avg_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return SentimentResult(
            text=f"[Aggregated {len(sentiments)} texts]",
            label=label,
            score=avg_score,
            confidence=avg_conf,
            model_name='aggregate',
            probabilities={}
        )

    def _calculate_overall_sentiment(
        self,
        title_sentiment: Optional[SentimentResult],
        content_sentiment: Optional[SentimentResult]
    ) -> Optional[SentimentResult]:
        """
        Calculate overall sentiment from title and content.

        Title is weighted more heavily (40%) as it's more focused.
        """
        if title_sentiment is None and content_sentiment is None:
            return None

        if title_sentiment is None:
            return content_sentiment
        if content_sentiment is None:
            return title_sentiment

        # Weighted combination (title: 40%, content: 60%)
        title_weight = 0.4
        content_weight = 0.6

        combined_score = (
            title_sentiment.score * title_weight +
            content_sentiment.score * content_weight
        )
        combined_conf = (
            title_sentiment.confidence * title_weight +
            content_sentiment.confidence * content_weight
        )

        if combined_score > 0.1:
            label = 'positive'
        elif combined_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return SentimentResult(
            text="[Overall]",
            label=label,
            score=combined_score,
            confidence=combined_conf,
            model_name='combined',
            probabilities={},
            metadata={
                'title_score': title_sentiment.score,
                'content_score': content_sentiment.score,
                'models': title_sentiment.metadata.get('models', {})
            }
        )

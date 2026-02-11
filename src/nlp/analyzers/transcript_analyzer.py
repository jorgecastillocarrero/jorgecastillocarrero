"""
Transcript Analyzer for earnings call transcripts.
Analyzes sentiment by section with special attention to Q&A dynamics.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..models.base import SentimentResult
from ..processors.text_cleaner import TranscriptTextCleaner
from ..processors.chunker import TranscriptChunker
from ..processors.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class TranscriptAnalysisResult:
    """Result of earnings call transcript analysis."""
    # Identification
    symbol: str = ""
    year: int = 0
    quarter: str = ""
    earnings_date: Optional[datetime] = None

    # Overall sentiment
    overall_score: float = 0.0
    overall_label: str = "neutral"
    confidence: float = 0.0

    # Section sentiments
    prepared_remarks_score: Optional[float] = None
    qa_section_score: Optional[float] = None
    guidance_score: Optional[float] = None

    # Important metrics
    qa_prepared_delta: Optional[float] = None  # Reveals tension if negative
    ceo_sentiment: Optional[float] = None
    cfo_sentiment: Optional[float] = None
    analyst_sentiment: Optional[float] = None

    # Topics detected
    topics: List[Dict] = field(default_factory=list)

    # Extracted entities
    tickers_mentioned: List[str] = field(default_factory=list)
    metrics_mentioned: List[Dict] = field(default_factory=list)

    # Metadata
    num_segments: int = 0
    transcript_length: int = 0
    processed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'symbol': self.symbol,
            'year': self.year,
            'quarter': self.quarter,
            'earnings_date': self.earnings_date,
            'overall_score': self.overall_score,
            'prepared_remarks_score': self.prepared_remarks_score,
            'qa_section_score': self.qa_section_score,
            'guidance_score': self.guidance_score,
            'qa_prepared_delta': self.qa_prepared_delta,
            'topics': self.topics,
            'num_segments': self.num_segments,
            'processed_at': self.processed_at,
        }


class TranscriptAnalyzer:
    """
    Analyzer for earnings call transcripts.

    Provides detailed analysis including:
    - Section-by-section sentiment (prepared remarks vs Q&A)
    - Speaker-level sentiment (CEO, CFO, analysts)
    - Topic detection
    - Q&A tension detection (delta between prepared and Q&A)
    """

    # Section markers
    QA_MARKERS = [
        r'question.?and.?answer',
        r'q\s*&\s*a\s+session',
        r'operator\s*:.*question',
        r'we\s+will\s+now\s+begin.*question',
    ]

    GUIDANCE_MARKERS = [
        r'guidance',
        r'outlook',
        r'looking\s+ahead',
        r'for\s+the\s+(?:next|coming)\s+(?:quarter|year)',
        r'we\s+expect',
        r'we\s+anticipate',
    ]

    # Topic keywords
    TOPIC_KEYWORDS = {
        'growth': ['growth', 'expand', 'increase', 'accelerat'],
        'margins': ['margin', 'profitability', 'cost', 'expense'],
        'guidance': ['guidance', 'outlook', 'forecast', 'expect'],
        'competition': ['competition', 'competitive', 'market share'],
        'innovation': ['innovation', 'new product', 'r&d', 'research'],
        'supply_chain': ['supply chain', 'inventory', 'shipping', 'logistics'],
        'macro': ['inflation', 'interest rate', 'recession', 'economy'],
        'ai_tech': ['artificial intelligence', 'ai', 'machine learning', 'automation'],
    }

    def __init__(self, sentiment_service=None):
        """
        Initialize transcript analyzer.

        Args:
            sentiment_service: Optional sentiment service instance
        """
        self._sentiment_service = sentiment_service
        self._cleaner = TranscriptTextCleaner()
        self._chunker = TranscriptChunker()
        self._extractor = EntityExtractor()

    def _get_sentiment_service(self):
        """Get or create sentiment service."""
        if self._sentiment_service is None:
            from ..services.sentiment_service import get_sentiment_service
            self._sentiment_service = get_sentiment_service()
        return self._sentiment_service

    def analyze(
        self,
        transcript: str,
        symbol: str = "",
        year: int = 0,
        quarter: str = "",
        earnings_date: Optional[datetime] = None
    ) -> TranscriptAnalysisResult:
        """
        Analyze an earnings call transcript.

        Args:
            transcript: Full transcript text
            symbol: Stock symbol
            year: Year of earnings
            quarter: Quarter (Q1, Q2, Q3, Q4)
            earnings_date: Date of earnings call

        Returns:
            TranscriptAnalysisResult
        """
        result = TranscriptAnalysisResult(
            symbol=symbol,
            year=year,
            quarter=quarter,
            earnings_date=earnings_date,
            transcript_length=len(transcript)
        )

        try:
            # Clean transcript
            clean_transcript = self._cleaner.clean(transcript)

            # Extract sections
            sections = self._extract_sections(clean_transcript)

            # Extract entities
            entities = self._extractor.extract(transcript)
            result.tickers_mentioned = entities.tickers
            result.metrics_mentioned = entities.metrics

            # Detect topics
            result.topics = self._detect_topics(clean_transcript)

            # Get sentiment service
            service = self._get_sentiment_service()

            # Analyze each section
            section_scores = {}

            for section_name, section_text in sections.items():
                if section_text:
                    chunks = self._chunker.chunk(section_text)
                    result.num_segments += len(chunks)

                    if chunks:
                        chunk_texts = [c.text for c in chunks]
                        sentiments = service.analyze_batch(chunk_texts)
                        section_score = self._aggregate_sentiments(sentiments)
                        section_scores[section_name] = section_score

            # Set section scores
            result.prepared_remarks_score = section_scores.get('prepared_remarks')
            result.qa_section_score = section_scores.get('qa')
            result.guidance_score = section_scores.get('guidance')

            # Calculate Q&A delta (reveals tension)
            if (result.prepared_remarks_score is not None and
                result.qa_section_score is not None):
                result.qa_prepared_delta = (
                    result.qa_section_score - result.prepared_remarks_score
                )

            # Analyze by speaker
            speaker_sentiments = self._analyze_by_speaker(clean_transcript, service)
            result.ceo_sentiment = speaker_sentiments.get('ceo')
            result.cfo_sentiment = speaker_sentiments.get('cfo')
            result.analyst_sentiment = speaker_sentiments.get('analyst')

            # Calculate overall score
            result.overall_score = self._calculate_overall_score(section_scores)

            if result.overall_score > 0.1:
                result.overall_label = 'positive'
            elif result.overall_score < -0.1:
                result.overall_label = 'negative'
            else:
                result.overall_label = 'neutral'

            # Confidence based on amount of text analyzed
            result.confidence = min(1.0, result.num_segments / 10)

        except Exception as e:
            logger.error(f"Error analyzing transcript {symbol} {year} {quarter}: {e}")

        return result

    def _extract_sections(self, transcript: str) -> Dict[str, str]:
        """
        Extract sections from transcript.

        Returns dict with:
        - prepared_remarks
        - qa
        - guidance
        """
        sections = {
            'prepared_remarks': '',
            'qa': '',
            'guidance': ''
        }

        transcript_lower = transcript.lower()

        # Find Q&A section start
        qa_start = len(transcript)
        for pattern in self.QA_MARKERS:
            match = re.search(pattern, transcript_lower)
            if match and match.start() < qa_start:
                qa_start = match.start()

        # Split prepared remarks and Q&A
        if qa_start < len(transcript):
            sections['prepared_remarks'] = transcript[:qa_start]
            sections['qa'] = transcript[qa_start:]
        else:
            sections['prepared_remarks'] = transcript

        # Find guidance within prepared remarks
        guidance_parts = []
        for pattern in self.GUIDANCE_MARKERS:
            for match in re.finditer(pattern, sections['prepared_remarks'].lower()):
                # Get surrounding context (100 chars before and after)
                start = max(0, match.start() - 100)
                end = min(len(sections['prepared_remarks']), match.end() + 300)
                guidance_parts.append(sections['prepared_remarks'][start:end])

        sections['guidance'] = ' '.join(guidance_parts)

        return sections

    def _detect_topics(self, text: str) -> List[Dict]:
        """
        Detect topics discussed in transcript.

        Returns list of topic dicts with name and count.
        """
        text_lower = text.lower()
        topics = []

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            count = sum(text_lower.count(kw.lower()) for kw in keywords)
            if count > 0:
                topics.append({
                    'topic': topic,
                    'mentions': count,
                    'relevance': min(1.0, count / 10)
                })

        # Sort by relevance
        topics.sort(key=lambda x: x['relevance'], reverse=True)
        return topics[:5]  # Top 5 topics

    def _analyze_by_speaker(
        self,
        transcript: str,
        service
    ) -> Dict[str, float]:
        """
        Analyze sentiment by speaker type.

        Identifies CEO, CFO, and analyst segments.
        """
        speaker_sentiments = {}

        # Speaker identification patterns
        speaker_patterns = {
            'ceo': [r'chief executive', r'\bceo\b', r'president'],
            'cfo': [r'chief financial', r'\bcfo\b', r'finance'],
            'analyst': [r'analyst', r'\bq:', r'question:'],
        }

        # Simple heuristic: find speaker labels and their following text
        speaker_regex = re.compile(r'\[([^\]]+)\]:\s*([^[]+)', re.DOTALL)

        for match in speaker_regex.finditer(transcript):
            speaker_label = match.group(1).lower()
            speaker_text = match.group(2).strip()

            if not speaker_text or len(speaker_text) < 20:
                continue

            for speaker_type, patterns in speaker_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, speaker_label, re.I):
                        # Analyze this segment
                        result = service.analyze(speaker_text[:1000])

                        if speaker_type in speaker_sentiments:
                            # Average with previous
                            speaker_sentiments[speaker_type] = (
                                speaker_sentiments[speaker_type] + result.score
                            ) / 2
                        else:
                            speaker_sentiments[speaker_type] = result.score
                        break

        return speaker_sentiments

    def _aggregate_sentiments(self, sentiments: List[SentimentResult]) -> float:
        """Aggregate sentiments to single score."""
        if not sentiments:
            return 0.0
        return sum(s.score for s in sentiments) / len(sentiments)

    def _calculate_overall_score(self, section_scores: Dict[str, float]) -> float:
        """
        Calculate overall transcript score.

        Weights:
        - Prepared remarks: 40%
        - Q&A: 40%
        - Guidance: 20%
        """
        weights = {
            'prepared_remarks': 0.4,
            'qa': 0.4,
            'guidance': 0.2
        }

        total_weight = 0.0
        weighted_score = 0.0

        for section, weight in weights.items():
            if section in section_scores and section_scores[section] is not None:
                weighted_score += section_scores[section] * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 0.0

    def analyze_batch(
        self,
        transcripts: List[Dict],
        batch_size: int = 32
    ) -> List[TranscriptAnalysisResult]:
        """
        Analyze multiple transcripts.

        Args:
            transcripts: List of transcript dicts
            batch_size: Batch size for processing

        Returns:
            List of TranscriptAnalysisResult
        """
        results = []

        for transcript_data in transcripts:
            result = self.analyze(
                transcript=transcript_data.get('content', ''),
                symbol=transcript_data.get('symbol', ''),
                year=transcript_data.get('year', 0),
                quarter=transcript_data.get('quarter', ''),
                earnings_date=transcript_data.get('date')
            )
            results.append(result)

        return results

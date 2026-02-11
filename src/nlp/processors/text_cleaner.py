"""
Text Cleaner for NLP preprocessing.
Handles cleaning, normalization, and preparation of financial texts.
"""

import re
import html
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for text cleaning."""
    lowercase: bool = False
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    normalize_numbers: bool = False
    preserve_tickers: bool = True
    min_length: int = 3


class TextCleaner:
    """
    Text cleaner for financial texts.

    Handles various preprocessing tasks while preserving
    financial-specific elements like tickers and numbers.
    """

    # Common patterns
    URL_PATTERN = re.compile(
        r'https?://\S+|www\.\S+',
        re.IGNORECASE
    )
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

    # Financial number patterns to preserve
    CURRENCY_PATTERN = re.compile(
        r'\$[\d,]+\.?\d*[MBK]?|\d+\.?\d*%|\d+\.?\d*x'
    )

    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize text cleaner.

        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """
        Clean text according to configuration.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        if self.config.remove_html:
            text = self._remove_html(text)

        # Remove URLs
        if self.config.remove_urls:
            text = self._remove_urls(text)

        # Remove emails
        if self.config.remove_emails:
            text = self._remove_emails(text)

        # Normalize whitespace
        if self.config.remove_extra_whitespace:
            text = self._normalize_whitespace(text)

        # Lowercase (preserving tickers if configured)
        if self.config.lowercase:
            text = self._lowercase_preserving_tickers(text)

        # Remove special characters (preserving important ones)
        if self.config.remove_special_chars:
            text = self._remove_special_chars(text)

        # Normalize numbers
        if self.config.normalize_numbers:
            text = self._normalize_numbers(text)

        # Final trim
        text = text.strip()

        # Check minimum length
        if len(text) < self.config.min_length:
            return ""

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        text = self.HTML_TAG_PATTERN.sub(' ', text)
        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.URL_PATTERN.sub(' ', text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.EMAIL_PATTERN.sub(' ', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to single spaces."""
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return self.WHITESPACE_PATTERN.sub(' ', text)

    def _lowercase_preserving_tickers(self, text: str) -> str:
        """Convert to lowercase while preserving ticker symbols."""
        if not self.config.preserve_tickers:
            return text.lower()

        # Find all tickers
        tickers = self.TICKER_PATTERN.findall(text)
        ticker_placeholders = {}

        # Replace tickers with placeholders
        for i, ticker in enumerate(tickers):
            placeholder = f"__TICKER_{i}__"
            ticker_placeholders[placeholder] = f"${ticker}"
            text = text.replace(f"${ticker}", placeholder, 1)

        # Lowercase
        text = text.lower()

        # Restore tickers
        for placeholder, ticker in ticker_placeholders.items():
            text = text.replace(placeholder.lower(), ticker)

        return text

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, preserving important ones."""
        # Preserve: letters, numbers, basic punctuation, $, %, .
        text = re.sub(r"[^\w\s\$%\.\,\-\'\"\:\;\!\?]", ' ', text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize large numbers to readable format."""
        def replace_number(match):
            num_str = match.group(0).replace(',', '')
            try:
                num = float(num_str)
                if num >= 1e12:
                    return f"{num/1e12:.1f}T"
                elif num >= 1e9:
                    return f"{num/1e9:.1f}B"
                elif num >= 1e6:
                    return f"{num/1e6:.1f}M"
                elif num >= 1e3:
                    return f"{num/1e3:.1f}K"
                return num_str
            except:
                return num_str

        # Match large numbers
        pattern = r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{4,}\b'
        return re.sub(pattern, replace_number, text)


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_html: bool = True,
    lowercase: bool = False
) -> str:
    """
    Convenience function for text cleaning.

    Args:
        text: Input text
        remove_urls: Remove URLs
        remove_html: Remove HTML
        lowercase: Convert to lowercase

    Returns:
        Cleaned text
    """
    config = CleaningConfig(
        remove_urls=remove_urls,
        remove_html=remove_html,
        lowercase=lowercase
    )
    cleaner = TextCleaner(config)
    return cleaner.clean(text)


# Specialized cleaners for different content types

class NewsTextCleaner(TextCleaner):
    """Cleaner specialized for news articles."""

    def __init__(self):
        config = CleaningConfig(
            lowercase=False,
            remove_urls=True,
            remove_emails=True,
            remove_html=True,
            remove_extra_whitespace=True,
            preserve_tickers=True
        )
        super().__init__(config)

    def clean(self, text: str) -> str:
        """Clean news text with additional processing."""
        text = super().clean(text)

        # Remove common news artifacts
        text = re.sub(r'\(Reuters\)|\(AP\)|\(Bloomberg\)', '', text)
        text = re.sub(r'^\s*[-\u2013\u2014]\s*', '', text)  # Leading dashes

        return text.strip()


class TranscriptTextCleaner(TextCleaner):
    """Cleaner specialized for earnings call transcripts."""

    SPEAKER_PATTERN = re.compile(
        r'^([A-Z][a-z]+ [A-Z][a-z]+|[A-Z\s]+):\s*',
        re.MULTILINE
    )

    def __init__(self):
        config = CleaningConfig(
            lowercase=False,
            remove_urls=True,
            remove_html=True,
            remove_extra_whitespace=True,
            preserve_tickers=True
        )
        super().__init__(config)

    def clean(self, text: str) -> str:
        """Clean transcript text."""
        text = super().clean(text)

        # Normalize speaker labels
        text = self._normalize_speakers(text)

        return text

    def _normalize_speakers(self, text: str) -> str:
        """Normalize speaker labels in transcript."""
        # Keep speaker labels but standardize format
        text = self.SPEAKER_PATTERN.sub(r'[\1]: ', text)
        return text

    def extract_sections(self, text: str) -> dict:
        """
        Extract sections from earnings call transcript.

        Returns dict with:
        - prepared_remarks: Opening statements
        - q_and_a: Q&A section
        - closing: Closing remarks
        """
        sections = {
            'prepared_remarks': '',
            'q_and_a': '',
            'closing': ''
        }

        # Common section markers
        qa_markers = [
            'question-and-answer',
            'questions and answers',
            'q&a',
            'operator:',
        ]

        text_lower = text.lower()
        qa_start = len(text)

        for marker in qa_markers:
            idx = text_lower.find(marker)
            if idx != -1 and idx < qa_start:
                qa_start = idx

        if qa_start < len(text):
            sections['prepared_remarks'] = text[:qa_start].strip()
            sections['q_and_a'] = text[qa_start:].strip()
        else:
            sections['prepared_remarks'] = text

        return sections

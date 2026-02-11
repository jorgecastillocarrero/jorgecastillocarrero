"""
Entity Extractor for financial texts.
Extracts tickers, companies, people, and financial metrics.
"""

import re
import logging
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    text: str
    entity_type: str  # 'ticker', 'company', 'person', 'metric', 'money'
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    tickers: List[str]
    companies: List[str]
    people: List[str]
    metrics: List[Dict]
    money_amounts: List[str]
    all_entities: List[ExtractedEntity]


class EntityExtractor:
    """
    Entity extractor for financial texts.

    Extracts:
    - Stock tickers ($AAPL, MSFT)
    - Company names
    - Person names (CEO, CFO mentions)
    - Financial metrics (P/E, EPS, revenue)
    - Money amounts ($1.5B, 2.3 million)
    """

    # Patterns
    CASHTAG_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')
    TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')

    MONEY_PATTERN = re.compile(
        r'\$[\d,]+\.?\d*\s*(?:million|billion|trillion|M|B|T|K)?'
        r'|[\d,]+\.?\d*\s*(?:million|billion|trillion)\s*(?:dollars|USD)?'
        r'|[\d,]+\.?\d*%',
        re.IGNORECASE
    )

    METRIC_PATTERNS = {
        'eps': re.compile(r'(?:EPS|earnings per share)\s*(?:of\s*)?\$?[\d.]+', re.I),
        'pe_ratio': re.compile(r'(?:P/E|PE ratio|price[- ]to[- ]earnings)\s*(?:of\s*)?[\d.]+', re.I),
        'revenue': re.compile(r'revenue\s*(?:of\s*)?\$?[\d.,]+\s*(?:million|billion|M|B)?', re.I),
        'guidance': re.compile(r'guidance\s*(?:of\s*)?\$?[\d.,]+\s*(?:to\s*\$?[\d.,]+)?', re.I),
        'margin': re.compile(r'(?:gross|operating|net|profit)\s*margin\s*(?:of\s*)?[\d.]+%', re.I),
    }

    TITLE_PATTERNS = re.compile(
        r'\b(?:CEO|CFO|COO|CTO|President|Chairman|Director|'
        r'Chief\s+(?:Executive|Financial|Operating|Technology)\s+Officer)\b',
        re.IGNORECASE
    )

    # Common company suffixes
    COMPANY_SUFFIXES = re.compile(
        r'\b(\w+(?:\s+\w+)*)\s*(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|'
        r'Limited|LLC|PLC|Group|Holdings|Technologies|'
        r'Pharmaceuticals|Therapeutics|Systems|Networks)\b',
        re.IGNORECASE
    )

    # Known valid tickers (could be loaded from database)
    COMMON_WORDS = {
        'CEO', 'CFO', 'COO', 'CTO', 'IPO', 'ETF', 'SEC', 'FDA', 'FTC',
        'NYSE', 'NASDAQ', 'GDP', 'CPI', 'PMI', 'THE', 'FOR', 'AND',
        'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE',
        'OUR', 'OUT', 'ARE', 'HAS', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW',
        'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HIM',
        'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'USA', 'CEO', 'CFO'
    }

    def __init__(self, valid_tickers: Optional[Set[str]] = None):
        """
        Initialize entity extractor.

        Args:
            valid_tickers: Set of known valid ticker symbols
        """
        self.valid_tickers = valid_tickers or set()

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all entities from text.

        Args:
            text: Input text

        Returns:
            ExtractionResult with all entities
        """
        all_entities = []

        # Extract tickers
        tickers = self._extract_tickers(text)
        for ticker, pos in tickers:
            all_entities.append(ExtractedEntity(
                text=ticker,
                entity_type='ticker',
                start_pos=pos[0],
                end_pos=pos[1],
                normalized=ticker.upper()
            ))

        # Extract money amounts
        money = self._extract_money(text)
        for amount, pos in money:
            all_entities.append(ExtractedEntity(
                text=amount,
                entity_type='money',
                start_pos=pos[0],
                end_pos=pos[1]
            ))

        # Extract metrics
        metrics = self._extract_metrics(text)
        for metric_type, value, pos in metrics:
            all_entities.append(ExtractedEntity(
                text=value,
                entity_type='metric',
                start_pos=pos[0],
                end_pos=pos[1],
                metadata={'metric_type': metric_type}
            ))

        # Extract companies
        companies = self._extract_companies(text)

        # Extract people/titles
        people = self._extract_people(text)

        return ExtractionResult(
            tickers=[e.text for e in all_entities if e.entity_type == 'ticker'],
            companies=companies,
            people=people,
            metrics=[{'type': e.metadata.get('metric_type'), 'value': e.text}
                    for e in all_entities if e.entity_type == 'metric'],
            money_amounts=[e.text for e in all_entities if e.entity_type == 'money'],
            all_entities=all_entities
        )

    def _extract_tickers(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Extract stock ticker symbols."""
        tickers = []

        # First, find cashtags (highest confidence)
        for match in self.CASHTAG_PATTERN.finditer(text):
            ticker = match.group(1)
            tickers.append((ticker, (match.start(), match.end())))

        # Then find potential tickers (need validation)
        for match in self.TICKER_PATTERN.finditer(text):
            ticker = match.group(1)

            # Skip common words
            if ticker in self.COMMON_WORDS:
                continue

            # Validate against known tickers if available
            if self.valid_tickers:
                if ticker in self.valid_tickers:
                    tickers.append((ticker, (match.start(), match.end())))
            else:
                # Heuristic: 2-4 letter uppercase likely ticker
                if 2 <= len(ticker) <= 4:
                    tickers.append((ticker, (match.start(), match.end())))

        # Deduplicate
        seen = set()
        unique_tickers = []
        for ticker, pos in tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append((ticker, pos))

        return unique_tickers

    def _extract_money(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Extract money amounts."""
        results = []
        for match in self.MONEY_PATTERN.finditer(text):
            results.append((match.group(0), (match.start(), match.end())))
        return results

    def _extract_metrics(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """Extract financial metrics."""
        results = []
        for metric_type, pattern in self.METRIC_PATTERNS.items():
            for match in pattern.finditer(text):
                results.append((metric_type, match.group(0), (match.start(), match.end())))
        return results

    def _extract_companies(self, text: str) -> List[str]:
        """Extract company names."""
        companies = []
        for match in self.COMPANY_SUFFIXES.finditer(text):
            company = match.group(0).strip()
            if len(company) > 3:
                companies.append(company)
        return list(set(companies))

    def _extract_people(self, text: str) -> List[str]:
        """Extract people mentions (by title)."""
        people = []

        for match in self.TITLE_PATTERNS.finditer(text):
            # Try to get name before or after title
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            # Look for name pattern around title
            title = match.group(0)
            people.append(title)

        return list(set(people))

    def extract_tickers_only(self, text: str) -> List[str]:
        """
        Extract only ticker symbols.

        Args:
            text: Input text

        Returns:
            List of ticker symbols
        """
        tickers = self._extract_tickers(text)
        return list(set([t[0] for t in tickers]))


def extract_tickers(text: str) -> List[str]:
    """
    Convenience function to extract tickers from text.

    Args:
        text: Input text

    Returns:
        List of ticker symbols
    """
    extractor = EntityExtractor()
    return extractor.extract_tickers_only(text)


class FinancialEntityLinker:
    """
    Links extracted entities to database records.

    Resolves ticker symbols and company names to
    canonical identifiers in the database.
    """

    def __init__(self, db_connection=None):
        """
        Initialize entity linker.

        Args:
            db_connection: Database connection for lookups
        """
        self.db = db_connection
        self._ticker_cache = {}
        self._company_cache = {}

    def link_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Link ticker to database record.

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with symbol info or None
        """
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]

        if self.db is None:
            return None

        # Query database
        try:
            from sqlalchemy import text
            result = self.db.execute(
                text("SELECT * FROM fmp_symbols WHERE symbol = :sym"),
                {'sym': ticker}
            ).fetchone()

            if result:
                info = dict(result._mapping)
                self._ticker_cache[ticker] = info
                return info
        except Exception as e:
            logger.warning(f"Error linking ticker {ticker}: {e}")

        return None

    def link_company(self, company_name: str) -> Optional[Dict]:
        """
        Link company name to database record.

        Args:
            company_name: Company name

        Returns:
            Dict with company info or None
        """
        if company_name in self._company_cache:
            return self._company_cache[company_name]

        if self.db is None:
            return None

        # Query database with fuzzy matching
        try:
            from sqlalchemy import text
            result = self.db.execute(
                text("""
                    SELECT * FROM fmp_profiles
                    WHERE company_name ILIKE :name
                    LIMIT 1
                """),
                {'name': f'%{company_name}%'}
            ).fetchone()

            if result:
                info = dict(result._mapping)
                self._company_cache[company_name] = info
                return info
        except Exception as e:
            logger.warning(f"Error linking company {company_name}: {e}")

        return None

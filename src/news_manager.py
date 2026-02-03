"""
News Manager Module for Financial AI Assistant
Handles storage, retrieval, and daily updates of financial news.

Sources:
- GDELT (free, historical)
- NewsAPI (recent, requires key)
- Finnhub (market news, requires key)
- Newspaper3k (article scraping)
"""

import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Container for a news article"""
    title: str
    source: str
    published_at: datetime
    url: str = None
    symbol: str = None
    summary: str = None
    content: str = None
    category: str = None
    sentiment: str = None
    relevance: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'title': self.title,
            'summary': self.summary,
            'content': self.content,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'category': self.category,
            'sentiment': self.sentiment,
            'relevance': self.relevance
        }


class NewsManager:
    """
    Manager for financial news storage and retrieval.
    """

    # Categories for classification
    CATEGORIES = [
        'earnings', 'merger', 'ipo', 'dividend', 'split',
        'crash', 'rally', 'fed', 'inflation', 'recession',
        'regulation', 'scandal', 'product', 'ceo', 'layoffs',
        'general'
    ]

    def __init__(self, db_path: str = "data/financial_data.db"):
        self.db_path = db_path
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    # =========================================================================
    # STORAGE
    # =========================================================================

    def save_article(self, article: NewsArticle) -> bool:
        """Save a single article to database"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news_history
                (symbol, title, summary, content, source, url, published_at,
                 category, sentiment, relevance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.symbol,
                article.title,
                article.summary,
                article.content,
                article.source,
                article.url,
                article.published_at.isoformat() if article.published_at else None,
                article.category,
                article.sentiment,
                article.relevance
            ))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error saving article: {e}")
            return False
        finally:
            conn.close()

    def save_articles(self, articles: List[NewsArticle]) -> Dict[str, int]:
        """Save multiple articles to database"""
        conn = self._get_connection()
        cursor = conn.cursor()

        saved = 0
        duplicates = 0
        errors = 0

        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_history
                    (symbol, title, summary, content, source, url, published_at,
                     category, sentiment, relevance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.symbol,
                    article.title,
                    article.summary,
                    article.content,
                    article.source,
                    article.url,
                    article.published_at.isoformat() if article.published_at else None,
                    article.category,
                    article.sentiment,
                    article.relevance
                ))
                if cursor.rowcount > 0:
                    saved += 1
                else:
                    duplicates += 1
            except Exception as e:
                errors += 1
                logger.error(f"Error saving article: {e}")

        conn.commit()
        conn.close()

        return {'saved': saved, 'duplicates': duplicates, 'errors': errors}

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_news(self, symbol: str = None, category: str = None,
                 start_date: str = None, end_date: str = None,
                 limit: int = 50) -> pd.DataFrame:
        """
        Get news from database.

        Args:
            symbol: Filter by symbol (None = all)
            category: Filter by category
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max records to return

        Returns:
            DataFrame with news articles
        """
        conn = self._get_connection()

        query = "SELECT * FROM news_history WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        if category:
            query += " AND category = ?"
            params.append(category)

        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date)

        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date)

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def search_news(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Search news by keyword in title/summary"""
        conn = self._get_connection()

        sql = """
            SELECT * FROM news_history
            WHERE title LIKE ? OR summary LIKE ?
            ORDER BY published_at DESC
            LIMIT ?
        """
        search_term = f"%{query}%"
        df = pd.read_sql_query(sql, conn, params=[search_term, search_term, limit])
        conn.close()

        return df

    def get_market_events(self, event_type: str = 'crash',
                          limit: int = 20) -> pd.DataFrame:
        """Get major market events (crashes, rallies, etc)"""
        return self.get_news(category=event_type, limit=limit)

    def get_news_summary(self, symbol: str = None, days: int = 30) -> str:
        """Generate text summary of recent news"""
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = self.get_news(symbol=symbol, start_date=start_date, limit=20)

        if df.empty:
            return f"No hay noticias recientes para {symbol or 'el mercado'}"

        lines = [f"=== NOTICIAS {'DE ' + symbol if symbol else 'DEL MERCADO'} ==="]
        lines.append(f"Ultimos {days} dias | {len(df)} articulos\n")

        for _, row in df.iterrows():
            date = row['published_at'][:10] if row['published_at'] else 'N/A'
            sentiment = f"[{row['sentiment']}]" if row['sentiment'] else ""
            lines.append(f"[{date}] {row['source']}: {row['title'][:70]}... {sentiment}")

        return "\n".join(lines)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get news database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM news_history")
        stats['total_articles'] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(published_at), MAX(published_at) FROM news_history")
        row = cursor.fetchone()
        stats['date_range'] = {'from': row[0], 'to': row[1]}

        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM news_history
            GROUP BY source
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_sources'] = cursor.fetchall()

        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM news_history
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        """)
        stats['by_category'] = cursor.fetchall()

        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM news_history
            WHERE symbol IS NOT NULL
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 20
        """)
        stats['top_symbols'] = cursor.fetchall()

        conn.close()
        return stats

    def stats_summary(self) -> str:
        """Generate text summary of database stats"""
        stats = self.get_stats()

        lines = [
            "=== ESTADISTICAS DE NOTICIAS ===",
            f"Total articulos: {stats['total_articles']:,}",
            f"Rango: {stats['date_range']['from']} a {stats['date_range']['to']}",
            "",
            "TOP FUENTES:"
        ]

        for source, count in stats['top_sources'][:5]:
            lines.append(f"  {source}: {count:,}")

        if stats['by_category']:
            lines.append("\nPOR CATEGORIA:")
            for cat, count in stats['by_category'][:5]:
                lines.append(f"  {cat}: {count:,}")

        return "\n".join(lines)

    # =========================================================================
    # DATA FETCHING - GDELT (FREE)
    # =========================================================================

    def fetch_gdelt(self, query: str, max_records: int = 100) -> List[NewsArticle]:
        """
        Fetch news from GDELT (free, unlimited history).

        Args:
            query: Search query
            max_records: Max articles to fetch

        Returns:
            List of NewsArticle
        """
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': query,
            'mode': 'ArtList',
            'maxrecords': max_records,
            'format': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            articles = []
            for item in data.get('articles', []):
                try:
                    pub_date = datetime.strptime(
                        item.get('seendate', '')[:14], '%Y%m%d%H%M%S'
                    ) if item.get('seendate') else None
                except:
                    pub_date = None

                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    source=item.get('domain', 'GDELT'),
                    published_at=pub_date,
                    url=item.get('url', ''),
                    summary=item.get('title', ''),
                    category=self._detect_category(item.get('title', ''))
                ))

            return articles

        except Exception as e:
            logger.error(f"GDELT fetch error: {e}")
            return []

    # =========================================================================
    # DATA FETCHING - NEWSAPI (RECENT, REQUIRES KEY)
    # =========================================================================

    def fetch_newsapi(self, query: str = None, symbol: str = None,
                      days: int = 7) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI (last 30 days max for free tier).

        Args:
            query: Search query
            symbol: Stock symbol
            days: Days back to search

        Returns:
            List of NewsArticle
        """
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not configured")
            return []

        if symbol:
            query = f"{symbol} stock"
        elif not query:
            query = "stock market"

        from_date = (datetime.now() - timedelta(days=min(days, 30))).strftime('%Y-%m-%d')

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.news_api_key,
            'pageSize': 100
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            articles = []
            if data.get('status') == 'ok':
                for item in data.get('articles', []):
                    try:
                        pub_date = datetime.fromisoformat(
                            item['publishedAt'].replace('Z', '+00:00')
                        ) if item.get('publishedAt') else None
                    except:
                        pub_date = None

                    articles.append(NewsArticle(
                        title=item.get('title', ''),
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        published_at=pub_date,
                        url=item.get('url', ''),
                        summary=item.get('description', ''),
                        content=item.get('content', ''),
                        symbol=symbol.upper() if symbol else None,
                        category=self._detect_category(item.get('title', ''))
                    ))

            return articles

        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _detect_category(self, text: str) -> str:
        """Auto-detect category from text"""
        text = text.lower()

        keywords = {
            'earnings': ['earnings', 'quarterly', 'profit', 'revenue', 'eps'],
            'crash': ['crash', 'plunge', 'collapse', 'crisis', 'panic', 'selloff'],
            'rally': ['rally', 'surge', 'soar', 'record high', 'bull'],
            'fed': ['fed', 'federal reserve', 'interest rate', 'powell', 'fomc'],
            'inflation': ['inflation', 'cpi', 'prices', 'cost of living'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
            'ipo': ['ipo', 'initial public offering', 'goes public', 'debut'],
            'dividend': ['dividend', 'payout', 'yield'],
            'layoffs': ['layoff', 'job cut', 'workforce reduction', 'firing'],
            'ceo': ['ceo', 'chief executive', 'leadership', 'resign'],
            'recession': ['recession', 'downturn', 'economic slowdown'],
            'regulation': ['sec', 'regulation', 'lawsuit', 'fine', 'compliance']
        }

        for category, words in keywords.items():
            if any(word in text for word in words):
                return category

        return 'general'

    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        text = text.lower()

        positive = ['surge', 'rally', 'gain', 'rise', 'profit', 'growth',
                    'beat', 'strong', 'positive', 'up', 'record', 'boom']
        negative = ['crash', 'fall', 'drop', 'loss', 'decline', 'plunge',
                    'miss', 'weak', 'negative', 'down', 'crisis', 'fear']

        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

    # =========================================================================
    # DAILY UPDATE
    # =========================================================================

    def run_daily_update(self, symbols: List[str] = None) -> Dict:
        """
        Run daily news update.

        Args:
            symbols: List of symbols to fetch news for

        Returns:
            Update results
        """
        results = {'gdelt': 0, 'newsapi': 0, 'total_saved': 0}

        # 1. Fetch general market news from GDELT
        logger.info("Fetching GDELT news...")
        gdelt_articles = self.fetch_gdelt("stock market OR financial markets", 50)
        if gdelt_articles:
            save_result = self.save_articles(gdelt_articles)
            results['gdelt'] = save_result['saved']

        # 2. Fetch from NewsAPI if key available
        if self.news_api_key:
            logger.info("Fetching NewsAPI news...")
            newsapi_articles = self.fetch_newsapi(query="stock market financial", days=1)
            if newsapi_articles:
                save_result = self.save_articles(newsapi_articles)
                results['newsapi'] = save_result['saved']

        # 3. Fetch for specific symbols
        if symbols:
            for symbol in symbols[:20]:  # Limit to avoid rate limits
                articles = self.fetch_gdelt(f"{symbol} stock", 10)
                for a in articles:
                    a.symbol = symbol.upper()
                if articles:
                    self.save_articles(articles)
                    results['total_saved'] += len(articles)

        results['total_saved'] += results['gdelt'] + results['newsapi']
        logger.info(f"Daily news update: {results['total_saved']} articles saved")

        return results


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = NewsManager()

    print("=== News Manager Test ===\n")

    # Stats
    print(manager.stats_summary())

    # Fetch some GDELT news
    print("\n--- Fetching GDELT news ---")
    articles = manager.fetch_gdelt("stock market crash", max_records=5)
    print(f"Fetched {len(articles)} articles")
    for a in articles[:3]:
        print(f"  - {a.title[:60]}...")

    # Save them
    if articles:
        result = manager.save_articles(articles)
        print(f"Saved: {result}")

    # Show updated stats
    print("\n" + manager.stats_summary())

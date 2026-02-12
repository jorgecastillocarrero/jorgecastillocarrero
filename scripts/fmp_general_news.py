"""
FMP General News Downloader.
Downloads general market news from FMP API.
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FMP_API_KEY = "PzRngOxBgNBSIhxbMOrOOAWjVZcna5Yf"
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
BASE_URL = "https://financialmodelingprep.com/stable/news/general-latest"

# Connection settings
MAX_CONCURRENT = 5
PAGE_SIZE = 100


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def create_table():
    """Create the general news table if not exists."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fmp_general_news (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20),
            title TEXT NOT NULL,
            text TEXT,
            published_date TIMESTAMP NOT NULL,
            publisher VARCHAR(255),
            site VARCHAR(255),
            url TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(url)
        );

        CREATE INDEX IF NOT EXISTS idx_general_news_published
        ON fmp_general_news(published_date DESC);

        CREATE INDEX IF NOT EXISTS idx_general_news_symbol
        ON fmp_general_news(symbol);
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Table fmp_general_news created/verified")


def get_existing_urls() -> set:
    """Get set of existing URLs to avoid duplicates."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT url FROM fmp_general_news")
    urls = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return urls


async def fetch_page(session: aiohttp.ClientSession, page: int) -> list:
    """Fetch a single page of news."""
    url = f"{BASE_URL}?page={page}&limit={PAGE_SIZE}&apikey={FMP_API_KEY}"

    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data if data else []
            else:
                logger.warning(f"Page {page}: Status {response.status}")
                return []
    except Exception as e:
        logger.error(f"Page {page}: Error {e}")
        return []


def insert_news(news_list: list, existing_urls: set) -> int:
    """Insert news into database, skipping duplicates."""
    if not news_list:
        return 0

    # Filter out existing URLs
    new_items = [n for n in news_list if n.get('url') not in existing_urls]

    if not new_items:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()

    values = []
    for item in new_items:
        try:
            pub_date = datetime.strptime(
                item.get('publishedDate', ''),
                '%Y-%m-%d %H:%M:%S'
            )
        except:
            pub_date = datetime.now()

        values.append((
            item.get('symbol'),
            item.get('title', '')[:2000],
            item.get('text'),
            pub_date,
            item.get('publisher'),
            item.get('site'),
            item.get('url'),
            item.get('image')
        ))

    try:
        execute_values(
            cur,
            """
            INSERT INTO fmp_general_news
            (symbol, title, text, published_date, publisher, site, url, image_url)
            VALUES %s
            ON CONFLICT (url) DO NOTHING
            """,
            values
        )
        inserted = cur.rowcount
        conn.commit()
    except Exception as e:
        logger.error(f"Insert error: {e}")
        conn.rollback()
        inserted = 0

    cur.close()
    conn.close()

    # Add new URLs to existing set
    for item in new_items:
        existing_urls.add(item.get('url'))

    return inserted


async def download_all_news():
    """Download all general news from FMP."""
    logger.info("Starting FMP General News download...")

    # Create table
    create_table()

    # Get existing URLs
    existing_urls = get_existing_urls()
    logger.info(f"Found {len(existing_urls)} existing news in database")

    total_downloaded = 0
    total_inserted = 0
    page = 0

    async with aiohttp.ClientSession() as session:
        while True:
            # Fetch batch of pages
            tasks = [
                fetch_page(session, p)
                for p in range(page, page + MAX_CONCURRENT)
            ]
            results = await asyncio.gather(*tasks)

            # Process results
            batch_empty = True
            for i, news_list in enumerate(results):
                if news_list:
                    batch_empty = False
                    total_downloaded += len(news_list)
                    inserted = insert_news(news_list, existing_urls)
                    total_inserted += inserted

            if batch_empty:
                logger.info(f"No more data at page {page}")
                break

            page += MAX_CONCURRENT
            logger.info(
                f"Progress: Downloaded {total_downloaded}, "
                f"Inserted {total_inserted} (page {page})"
            )

            # Small delay to be nice to API
            await asyncio.sleep(0.5)

    return {
        'downloaded': total_downloaded,
        'inserted': total_inserted,
        'existing': len(existing_urls) - total_inserted
    }


def get_stats():
    """Get statistics about downloaded news."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total,
            MIN(published_date) as oldest,
            MAX(published_date) as newest,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT publisher) as unique_publishers
        FROM fmp_general_news
    """)

    row = cur.fetchone()

    cur.close()
    conn.close()

    return {
        'total': row[0],
        'oldest': row[1],
        'newest': row[2],
        'unique_symbols': row[3],
        'unique_publishers': row[4]
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print("\n=== FMP General News Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        results = asyncio.run(download_all_news())
        print("\n=== DOWNLOAD COMPLETE ===")
        print(f"  Downloaded: {results['downloaded']}")
        print(f"  Inserted: {results['inserted']}")
        print(f"  Already existed: {results['existing']}")

        stats = get_stats()
        print("\n=== DATABASE STATS ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

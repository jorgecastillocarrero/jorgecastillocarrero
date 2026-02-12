"""
GDELT 1.0 Economic Events Downloader.
Downloads economic events from 1920 to present from BigQuery to PostgreSQL.
"""

import os
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/gcloud-key.json'

from google.cloud import bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"
START_YEAR = 1920
END_YEAR = 2026

# GDELT Event Codes for economic events (04 = Economy)
# 040 = Economic cooperation
# 041 = Provide aid
# 042 = Provide economic support (interest rates, etc.)
# 043 = Trade/Currency
# 044 = Economic pressure


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def download_year(client, year: int) -> int:
    """Download economic events for a specific year."""

    query = f"""
    SELECT
        SQLDATE,
        Actor1Name,
        Actor1CountryCode,
        Actor1Type1Code,
        Actor2Name,
        Actor2CountryCode,
        Actor2Type1Code,
        EventCode,
        EventBaseCode,
        GoldsteinScale,
        AvgTone,
        NumMentions,
        NumSources,
        NumArticles,
        SOURCEURL
    FROM `gdelt-bq.full.events`
    WHERE EventRootCode = '04'
      AND SQLDATE >= {year}0101
      AND SQLDATE < {year + 1}0101
    """

    logger.info(f"Querying year {year}...")

    try:
        result = client.query(query).result()

        rows = []
        for row in result:
            # Parse date from YYYYMMDD integer
            date_int = row.SQLDATE
            if date_int:
                try:
                    event_date = datetime.strptime(str(date_int), '%Y%m%d').date()
                except:
                    continue
            else:
                continue

            rows.append((
                event_date,
                row.Actor1Name[:255] if row.Actor1Name else None,
                row.Actor1CountryCode[:10] if row.Actor1CountryCode else None,
                row.Actor1Type1Code[:50] if row.Actor1Type1Code else None,
                row.Actor2Name[:255] if row.Actor2Name else None,
                row.Actor2CountryCode[:10] if row.Actor2CountryCode else None,
                row.Actor2Type1Code[:50] if row.Actor2Type1Code else None,
                row.EventCode[:10] if row.EventCode else None,
                row.EventBaseCode[:10] if row.EventBaseCode else None,
                row.GoldsteinScale,
                row.AvgTone,
                row.NumMentions,
                row.NumSources,
                row.NumArticles,
                row.SOURCEURL[:2000] if row.SOURCEURL else None
            ))

        if rows:
            insert_events(rows)

        logger.info(f"Year {year}: {len(rows):,} events")
        return len(rows)

    except Exception as e:
        logger.error(f"Year {year}: Error - {e}")
        return 0


def insert_events(rows: list):
    """Insert events into PostgreSQL."""
    conn = get_db_connection()
    cur = conn.cursor()

    execute_values(
        cur,
        """
        INSERT INTO gdelt_events
        (event_date, actor1_name, actor1_country, actor1_type,
         actor2_name, actor2_country, actor2_type,
         event_code, event_description, goldstein_scale, avg_tone,
         num_mentions, num_sources, num_articles, source_url)
        VALUES %s
        """,
        rows,
        page_size=10000
    )

    conn.commit()
    cur.close()
    conn.close()


def get_stats():
    """Get statistics about downloaded events."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total,
            MIN(event_date) as oldest,
            MAX(event_date) as newest,
            COUNT(DISTINCT event_code) as unique_codes,
            COUNT(DISTINCT actor1_country) as unique_countries
        FROM gdelt_events
    """)

    row = cur.fetchone()
    cur.close()
    conn.close()

    return {
        'total': row[0],
        'oldest': row[1],
        'newest': row[2],
        'unique_codes': row[3],
        'unique_countries': row[4]
    }


def main():
    """Main download function."""
    logger.info("Starting GDELT 1.0 Economic Events download...")
    logger.info(f"Period: {START_YEAR} - {END_YEAR}")

    client = bigquery.Client()

    total_events = 0

    for year in range(START_YEAR, END_YEAR + 1):
        count = download_year(client, year)
        total_events += count

        if count == 0 and year > 1980:
            # Might have hit quota
            logger.warning(f"No events for {year}, might be quota issue")

    logger.info(f"\n=== DOWNLOAD COMPLETE ===")
    logger.info(f"Total events: {total_events:,}")

    stats = get_stats()
    logger.info(f"\n=== DATABASE STATS ===")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print("\n=== GDELT Events Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with just one year
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/gcloud-key.json'
        client = bigquery.Client()
        count = download_year(client, 2024)
        print(f"Test download: {count} events from 2024")
    else:
        main()

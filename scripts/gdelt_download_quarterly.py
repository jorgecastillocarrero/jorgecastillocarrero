"""
GDELT 1.0 Economic Events Downloader - By Quarter.
Downloads economic events by quarter to avoid timeouts.
"""

import os
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
import sys

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

# Quarters to download
QUARTERS = [
    (2021, 1), (2021, 2), (2021, 3), (2021, 4),
    (2022, 1), (2022, 2), (2022, 3), (2022, 4),
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2), (2025, 3), (2025, 4),
    (2026, 1),  # Current quarter
]


def get_quarter_dates(year: int, quarter: int):
    """Get start and end dates for a quarter."""
    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3

    start_date = f"{year}{start_month:02d}01"

    # End date is first day of next quarter
    if quarter == 4:
        end_date = f"{year + 1}0101"
    else:
        end_date = f"{year}{end_month + 1:02d}01"

    return start_date, end_date


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def check_quarter_exists(year: int, quarter: int) -> int:
    """Check how many events exist for this quarter."""
    start_date, end_date = get_quarter_dates(year, quarter)

    # Convert to date format
    start = f"{year}-{(quarter-1)*3+1:02d}-01"
    if quarter == 4:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{quarter*3+1:02d}-01"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT COUNT(*) FROM gdelt_events
        WHERE event_date >= '{start}' AND event_date < '{end}'
    """)
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def download_quarter(client, year: int, quarter: int) -> int:
    """Download economic events for a specific quarter."""

    start_date, end_date = get_quarter_dates(year, quarter)

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
      AND SQLDATE >= {start_date}
      AND SQLDATE < {end_date}
    """

    logger.info(f"Querying {year} Q{quarter} ({start_date} - {end_date})...")

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

        logger.info(f"{year} Q{quarter}: {len(rows):,} events downloaded")
        return len(rows)

    except Exception as e:
        logger.error(f"{year} Q{quarter}: Error - {e}")
        return 0


def insert_events(rows: list):
    """Insert events into PostgreSQL in batches."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Insert in batches of 50000
    batch_size = 50000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
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
            batch,
            page_size=10000
        )
        conn.commit()
        logger.info(f"  Inserted batch {i//batch_size + 1} ({len(batch):,} rows)")

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
            MAX(event_date) as newest
        FROM gdelt_events
    """)

    row = cur.fetchone()
    cur.close()
    conn.close()

    return {
        'total': row[0],
        'oldest': row[1],
        'newest': row[2]
    }


def main():
    """Main download function."""
    logger.info("=" * 60)
    logger.info("GDELT 1.0 Economic Events - Quarterly Download")
    logger.info("=" * 60)

    client = bigquery.Client()

    total_events = 0
    skipped = 0

    for year, quarter in QUARTERS:
        # Check if quarter already downloaded
        existing = check_quarter_exists(year, quarter)
        if existing > 10000:
            logger.info(f"{year} Q{quarter}: Already has {existing:,} events, skipping")
            skipped += 1
            continue

        count = download_quarter(client, year, quarter)
        total_events += count

        if count == 0:
            logger.warning(f"{year} Q{quarter}: No events (might be future date or quota)")

    logger.info("=" * 60)
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"  New events: {total_events:,}")
    logger.info(f"  Quarters skipped: {skipped}")

    stats = get_stats()
    logger.info(f"\nDatabase totals:")
    logger.info(f"  Total events: {stats['total']:,}")
    logger.info(f"  Date range: {stats['oldest']} to {stats['newest']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print("\n=== GDELT Events Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif len(sys.argv) > 2:
        # Download specific quarter: script.py 2024 2
        year = int(sys.argv[1])
        quarter = int(sys.argv[2])
        client = bigquery.Client()
        count = download_quarter(client, year, quarter)
        print(f"Downloaded {count:,} events for {year} Q{quarter}")
    else:
        main()

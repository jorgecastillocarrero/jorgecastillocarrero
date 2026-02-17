"""
GDELT 1.0 Economic Events Downloader - Direct Download.
Downloads without checking existing data first (faster startup).
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


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(DB_URL)


def download_quarter(client, year: int, quarter: int) -> int:
    """Download economic events for a specific quarter."""

    # Calculate date range
    start_month = (quarter - 1) * 3 + 1
    start_date = f"{year}{start_month:02d}01"

    if quarter == 4:
        end_date = f"{year + 1}0101"
    else:
        end_date = f"{year}{(quarter * 3) + 1:02d}01"

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

    logger.info(f"Descargando {year} Q{quarter}...")

    try:
        result = client.query(query).result()

        rows = []
        for row in result:
            date_int = row.SQLDATE
            if not date_int:
                continue
            try:
                event_date = datetime.strptime(str(date_int), '%Y%m%d').date()
            except:
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
            logger.info(f"  {year} Q{quarter}: {len(rows):,} eventos insertados")
        else:
            logger.info(f"  {year} Q{quarter}: sin eventos")

        return len(rows)

    except Exception as e:
        logger.error(f"  {year} Q{quarter}: Error - {e}")
        return 0


def insert_events(rows: list):
    """Insert events into PostgreSQL in batches."""
    conn = get_db_connection()
    cur = conn.cursor()

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
            ON CONFLICT DO NOTHING
            """,
            batch,
            page_size=10000
        )
        conn.commit()

    cur.close()
    conn.close()


def main():
    """Main download function."""
    # Parse arguments
    if len(sys.argv) < 3:
        print("Uso: py -3 gdelt_download_direct.py <año> <trimestre>")
        print("Ejemplo: py -3 gdelt_download_direct.py 2021 1")
        print("\nO para descargar un rango:")
        print("  py -3 gdelt_download_direct.py 2021 1 2026 1")
        return

    start_year = int(sys.argv[1])
    start_quarter = int(sys.argv[2])

    if len(sys.argv) >= 5:
        end_year = int(sys.argv[3])
        end_quarter = int(sys.argv[4])
    else:
        end_year = start_year
        end_quarter = start_quarter

    logger.info("=" * 50)
    logger.info("GDELT Economic Events - Direct Download")
    logger.info("=" * 50)

    client = bigquery.Client()
    total = 0

    year, quarter = start_year, start_quarter
    while (year < end_year) or (year == end_year and quarter <= end_quarter):
        count = download_quarter(client, year, quarter)
        total += count

        # Next quarter
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1

    logger.info("=" * 50)
    logger.info(f"COMPLETADO: {total:,} eventos descargados")


if __name__ == "__main__":
    main()

"""
GDELT News Events Downloader via CSV files.
Downloads economic/financial events directly from GDELT public CSV files.
No BigQuery quota needed.
"""

import os
import logging
import requests
import zipfile
import io
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://fmp:fmp123@localhost:5433/fmp_data"

# GDELT 1.0 column names (58 columns)
GDELT_COLUMNS = [
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate',
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
    'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
    'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
    'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
    'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
    'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
    'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
    'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
    'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
    'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
    'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat',
    'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type',
    'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
    'ActionGeo_ADM2Code', 'ActionGeo_Lat', 'ActionGeo_Long',
    'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
]


def get_db_connection():
    return psycopg2.connect(DB_URL)


def download_gdelt_file(date_str: str) -> pd.DataFrame:
    """Download GDELT file for a specific date (YYYYMMDD format)."""
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            return None
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, sep='\t', header=None, names=GDELT_COLUMNS,
                               dtype=str, on_bad_lines='skip')
        return df
    except Exception as e:
        logger.debug(f"Error downloading {date_str}: {e}")
        return None


def filter_economic_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for economic events (EventRootCode = 04)."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Economic events: EventRootCode starts with '04'
    # Also include financial actors (BNK=Bank, BUS=Business, etc.)
    economic_mask = df['EventRootCode'].astype(str).str.startswith('04')

    # Also include events involving financial actors
    financial_actors = ['BNK', 'BUS', 'FIN', 'MNC']  # Bank, Business, Financial, Multinational Corp
    actor_mask = (
        df['Actor1Type1Code'].isin(financial_actors) |
        df['Actor2Type1Code'].isin(financial_actors)
    )

    return df[economic_mask | actor_mask].copy()


def process_and_insert(df: pd.DataFrame, date_str: str) -> int:
    """Process dataframe and insert into PostgreSQL."""
    if df is None or df.empty:
        return 0

    rows = []
    for _, row in df.iterrows():
        try:
            # Parse date
            day_val = row['Day']
            if pd.isna(day_val):
                continue
            event_date = datetime.strptime(str(int(float(day_val))), '%Y%m%d').date()

            # Extract values with safe handling
            def safe_str(val, max_len=255):
                if pd.isna(val):
                    return None
                return str(val)[:max_len]

            def safe_float(val):
                if pd.isna(val):
                    return None
                try:
                    return float(val)
                except:
                    return None

            def safe_int(val):
                if pd.isna(val):
                    return None
                try:
                    return int(float(val))
                except:
                    return None

            rows.append((
                event_date,
                safe_str(row['Actor1Name']),
                safe_str(row['Actor1CountryCode'], 10),
                safe_str(row['Actor1Type1Code'], 50),
                safe_str(row['Actor2Name']),
                safe_str(row['Actor2CountryCode'], 10),
                safe_str(row['Actor2Type1Code'], 50),
                safe_str(row['EventCode'], 10),
                safe_str(row['EventBaseCode'], 10),
                safe_float(row['GoldsteinScale']),
                safe_float(row['AvgTone']),
                safe_int(row['NumMentions']),
                safe_int(row['NumSources']),
                safe_int(row['NumArticles']),
                safe_str(row['SOURCEURL'], 2000)
            ))
        except Exception as e:
            continue

    if rows:
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
            page_size=5000
        )
        conn.commit()
        cur.close()
        conn.close()

    return len(rows)


def download_date_range(start_date: datetime, end_date: datetime, max_workers: int = 4):
    """Download GDELT data for a date range."""

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    total_events = 0
    processed = 0

    logger.info(f"Downloading {len(dates)} days from {start_date.date()} to {end_date.date()}")

    for date_str in dates:
        try:
            df = download_gdelt_file(date_str)
            if df is not None:
                filtered = filter_economic_events(df)
                count = process_and_insert(filtered, date_str)
                total_events += count
                if count > 0:
                    logger.info(f"{date_str}: {count:,} economic/financial events")

            processed += 1
            if processed % 30 == 0:
                logger.info(f"Progress: {processed}/{len(dates)} days, {total_events:,} total events")

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            continue

    return total_events


def get_last_date_in_db():
    """Get the most recent date in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(event_date) FROM gdelt_events")
    result = cur.fetchone()[0]
    cur.close()
    conn.close()
    return result


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
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_stats()
        print("\n=== GDELT Events Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "recent":
        # Download last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
    elif len(sys.argv) > 2:
        # Custom date range: script.py 2024-01-01 2024-12-31
        start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
    else:
        # Default: last year for financial news
        end_date = datetime.now()
        start_date = datetime(2020, 1, 1)  # From 2020 onwards

    logger.info("="*60)
    logger.info("GDELT Economic/Financial News Downloader (CSV)")
    logger.info("="*60)

    # Check for existing data
    last_date = get_last_date_in_db()
    if last_date:
        logger.info(f"Last date in DB: {last_date}")
        # Continue from last date
        start_date = datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)
        logger.info(f"Continuing from: {start_date.date()}")

    total = download_date_range(start_date, end_date)

    logger.info("="*60)
    logger.info(f"COMPLETE: {total:,} events downloaded")

    stats = get_stats()
    logger.info("\nDatabase stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()

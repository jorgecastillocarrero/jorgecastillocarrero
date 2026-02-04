"""
Script to migrate exchange rate data from SQLite to PostgreSQL.
Only migrates currency pairs (EURUSD=X, etc.) - much smaller than full price_history.

Usage:
    python scripts/migrate_exchange_rates.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from sqlalchemy import create_engine, text
import pandas as pd

# Currency symbols to migrate
CURRENCY_SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'CADEUR=X', 'CHFEUR=X'
]

def main():
    # Source: local SQLite
    source_url = f"sqlite:///{Path(__file__).parent.parent / 'data' / 'financial_data.db'}"

    # Target: PostgreSQL from command line, environment, or prompt
    target_url = None
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    if not target_url:
        target_url = os.environ.get('DATABASE_URL')
    if not target_url:
        target_url = input("Enter PostgreSQL DATABASE_URL: ").strip()

    print(f"Source: {source_url}")
    print(f"Target: {target_url[:50]}...")

    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)

    # Get symbol IDs for currency pairs from source
    placeholders = ', '.join([f"'{s}'" for s in CURRENCY_SYMBOLS])
    with source_engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT id, code FROM symbols
            WHERE code IN ({placeholders})
        """))
        source_symbols = {row[1]: row[0] for row in result.fetchall()}

    print(f"\nFound {len(source_symbols)} currency symbols in source: {list(source_symbols.keys())}")

    # Get symbol IDs from target (might be different)
    with target_engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT id, code FROM symbols
            WHERE code IN ({placeholders})
        """))
        target_symbols = {row[1]: row[0] for row in result.fetchall()}

    print(f"Found {len(target_symbols)} currency symbols in target: {list(target_symbols.keys())}")

    # Migrate price data for each symbol
    total_migrated = 0
    for symbol_code in CURRENCY_SYMBOLS:
        if symbol_code not in source_symbols:
            print(f"  {symbol_code}: not in source, skipping")
            continue
        if symbol_code not in target_symbols:
            print(f"  {symbol_code}: not in target, skipping")
            continue

        source_id = source_symbols[symbol_code]
        target_id = target_symbols[symbol_code]

        # Read price data from source
        df = pd.read_sql(f"""
            SELECT date, open, high, low, close, adjusted_close, volume
            FROM price_history
            WHERE symbol_id = {source_id}
        """, source_engine)

        if df.empty:
            print(f"  {symbol_code}: no data in source")
            continue

        # Update symbol_id to target ID
        df['symbol_id'] = target_id

        # Delete existing data in target for this symbol
        with target_engine.connect() as conn:
            conn.execute(text(f"DELETE FROM price_history WHERE symbol_id = {target_id}"))
            conn.commit()

        # Insert new data
        df.to_sql('price_history', target_engine, if_exists='append', index=False, method='multi', chunksize=500)

        print(f"  {symbol_code}: migrated {len(df)} rows")
        total_migrated += len(df)

    print(f"\n✓ Total migrated: {total_migrated} rows")

if __name__ == '__main__':
    main()

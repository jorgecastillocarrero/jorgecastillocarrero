"""
Continue migrating price_history from SQLite to PostgreSQL.
Only migrates symbols that are missing or incomplete.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
import pandas as pd

def main():
    source_url = f"sqlite:///{Path(__file__).parent.parent / 'data' / 'financial_data.db'}"
    target_url = sys.argv[1] if len(sys.argv) > 1 else None
    if not target_url:
        print("Usage: python continue_migration.py <postgresql_url>")
        return

    print(f"Source: {source_url}")
    print(f"Target: {target_url[:50]}...")

    source = create_engine(source_url)
    target = create_engine(target_url)

    # Get symbol counts from both databases
    print("\nAnalyzing differences...")

    with source.connect() as conn:
        result = conn.execute(text('''
            SELECT s.id, s.code, COUNT(p.id) as cnt
            FROM symbols s
            LEFT JOIN price_history p ON s.id = p.symbol_id
            GROUP BY s.id
        '''))
        source_counts = {r[1]: (r[0], r[2]) for r in result.fetchall()}

    with target.connect() as conn:
        result = conn.execute(text('''
            SELECT s.id, s.code, COUNT(p.id) as cnt
            FROM symbols s
            LEFT JOIN price_history p ON s.id = p.symbol_id
            GROUP BY s.id, s.code
        '''))
        target_counts = {r[1]: (r[0], r[2]) for r in result.fetchall()}

    # Find symbols needing migration
    to_migrate = []
    for code, (src_id, src_cnt) in source_counts.items():
        if code not in target_counts:
            continue
        tgt_id, tgt_cnt = target_counts[code]
        if src_cnt > tgt_cnt:
            to_migrate.append((code, src_id, tgt_id, src_cnt, tgt_cnt))

    to_migrate.sort(key=lambda x: -(x[3] - x[4]))

    total_to_migrate = sum(x[3] - x[4] for x in to_migrate)
    print(f"Found {len(to_migrate)} symbols with {total_to_migrate:,} rows to migrate\n")

    if not to_migrate:
        print("Nothing to migrate!")
        return

    # Migrate each symbol
    migrated_total = 0
    for i, (code, src_id, tgt_id, src_cnt, tgt_cnt) in enumerate(to_migrate):
        missing = src_cnt - tgt_cnt
        print(f"[{i+1}/{len(to_migrate)}] {code}: migrating {missing:,} rows...", end=" ", flush=True)

        try:
            # Read from source
            df = pd.read_sql(f"""
                SELECT date, open, high, low, close, adjusted_close, volume
                FROM price_history
                WHERE symbol_id = {src_id}
            """, source)

            if df.empty:
                print("(empty)")
                continue

            # Set target symbol_id
            df['symbol_id'] = tgt_id

            # Delete existing and insert all (simpler than incremental)
            with target.connect() as conn:
                conn.execute(text(f"DELETE FROM price_history WHERE symbol_id = {tgt_id}"))
                conn.commit()

            df.to_sql('price_history', target, if_exists='append', index=False, method='multi', chunksize=500)

            migrated_total += len(df)
            progress = (migrated_total / total_to_migrate) * 100
            print(f"OK ({len(df):,} rows) - {progress:.1f}%")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nMigration complete! Total migrated: {migrated_total:,} rows")

if __name__ == '__main__':
    main()

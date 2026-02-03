"""
Script to migrate data from SQLite to PostgreSQL.
Run this locally before deploying to production.

Usage:
    python scripts/migrate_to_postgres.py --source sqlite:///data/financial_data.db --target postgresql://user:pass@host:5432/db
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import pandas as pd


def get_all_tables(engine):
    """Get list of all tables in database."""
    inspector = inspect(engine)
    return inspector.get_table_names()


def migrate_table(source_engine, target_engine, table_name):
    """Migrate a single table from source to target."""
    print(f"  Migrating table: {table_name}...", end=" ")

    try:
        # Read all data from source
        df = pd.read_sql_table(table_name, source_engine)

        if df.empty:
            print("(empty)")
            return 0

        # Write to target (append mode, table should already exist)
        df.to_sql(
            table_name,
            target_engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )

        print(f"({len(df)} rows)")
        return len(df)

    except Exception as e:
        print(f"ERROR: {e}")
        return 0


def migrate_database(source_url: str, target_url: str):
    """Migrate all data from SQLite to PostgreSQL."""
    print("\n=== Database Migration Tool ===\n")

    # Create engines
    print(f"Source: {source_url}")
    print(f"Target: {target_url}\n")

    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)

    # Get tables from source
    tables = get_all_tables(source_engine)
    print(f"Found {len(tables)} tables to migrate\n")

    # Create tables in target (import models to register them)
    from src.database import Base
    Base.metadata.create_all(bind=target_engine)
    print("Target schema created\n")

    # Migrate each table
    total_rows = 0
    for table in tables:
        rows = migrate_table(source_engine, target_engine, table)
        total_rows += rows

    print(f"\n=== Migration Complete ===")
    print(f"Total rows migrated: {total_rows}")

    # Verify
    print("\n=== Verification ===")
    target_tables = get_all_tables(target_engine)
    print(f"Tables in target: {len(target_tables)}")

    with target_engine.connect() as conn:
        for table in target_tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table}: {count} rows")


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument(
        "--source",
        required=True,
        help="Source database URL (e.g., sqlite:///data/financial_data.db)"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target database URL (e.g., postgresql://user:pass@host:5432/db)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        source_engine = create_engine(args.source)
        tables = get_all_tables(source_engine)
        print(f"\nTables that would be migrated: {tables}")
    else:
        migrate_database(args.source, args.target)


if __name__ == "__main__":
    main()

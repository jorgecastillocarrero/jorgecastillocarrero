"""
Batch download script for large symbol lists.
Downloads historical prices and fundamentals from Yahoo Finance.
"""

import sys
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, '.')

from src.yahoo_downloader import YahooDownloader
from src.database import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_symbols(filepath: str) -> list[str]:
    """Load symbols from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_existing_symbols() -> set[str]:
    """Get symbols already in database."""
    db = get_db_manager()
    with db.get_session() as session:
        from src.database import Symbol
        existing = session.query(Symbol.code).all()
        return {s[0] for s in existing}


def batch_download_prices(
    symbols: list[str],
    batch_size: int = 50,
    skip_existing: bool = True,
    period: str = "max"
) -> dict:
    """
    Download prices for symbols in batches.

    Args:
        symbols: List of symbols to download
        batch_size: Number of symbols per batch
        skip_existing: Skip symbols already in database
        period: Historical data period

    Returns:
        Results summary
    """
    downloader = YahooDownloader()

    # Filter out existing symbols if requested
    if skip_existing:
        existing = get_existing_symbols()
        new_symbols = [s for s in symbols if s not in existing]
        logger.info(f"Skipping {len(symbols) - len(new_symbols)} existing symbols")
        symbols = new_symbols

    total = len(symbols)
    logger.info(f"Downloading prices for {total} symbols")

    results = {
        "total": total,
        "success": 0,
        "failed": 0,
        "total_records": 0,
        "errors": [],
    }

    start_time = time.time()

    for i, symbol in enumerate(symbols, 1):
        try:
            count = downloader.download_historical_prices(symbol, period=period)
            results["success"] += 1
            results["total_records"] += count

            if i % 10 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i}/{total} ({100*i/total:.1f}%) | "
                    f"Success: {results['success']} | "
                    f"Records: {results['total_records']:,} | "
                    f"ETA: {eta/60:.1f}min"
                )

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"symbol": symbol, "error": str(e)[:100]})
            logger.warning(f"{symbol}: Failed - {str(e)[:50]}")

        # Small delay to avoid rate limiting
        if i % 100 == 0:
            time.sleep(1)

    elapsed = time.time() - start_time
    logger.info(f"Prices download completed in {elapsed/60:.1f} minutes")

    return results


def batch_download_fundamentals(
    symbols: list[str],
    batch_size: int = 50,
) -> dict:
    """
    Download fundamentals for symbols.

    Args:
        symbols: List of symbols to download
        batch_size: Number of symbols per batch

    Returns:
        Results summary
    """
    downloader = YahooDownloader()

    # Get symbols that exist in database
    existing = get_existing_symbols()
    symbols = [s for s in symbols if s in existing]

    total = len(symbols)
    logger.info(f"Downloading fundamentals for {total} symbols")

    results = {
        "total": total,
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    start_time = time.time()

    for i, symbol in enumerate(symbols, 1):
        try:
            if downloader.download_fundamentals(symbol):
                results["success"] += 1
            else:
                results["failed"] += 1

            if i % 20 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i}/{total} ({100*i/total:.1f}%) | "
                    f"Success: {results['success']} | "
                    f"ETA: {eta/60:.1f}min"
                )

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"symbol": symbol, "error": str(e)[:100]})

        # Small delay to avoid rate limiting
        if i % 50 == 0:
            time.sleep(1)

    elapsed = time.time() - start_time
    logger.info(f"Fundamentals download completed in {elapsed/60:.1f} minutes")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch download financial data")
    parser.add_argument("--symbols-file", default="symbols_to_download.txt",
                        help="File with symbols to download")
    parser.add_argument("--prices", action="store_true",
                        help="Download historical prices")
    parser.add_argument("--fundamentals", action="store_true",
                        help="Download fundamentals")
    parser.add_argument("--all", action="store_true",
                        help="Download both prices and fundamentals")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip symbols already in database")
    parser.add_argument("--period", default="max",
                        help="Historical data period (max, 10y, 5y, 2y, 1y)")

    args = parser.parse_args()

    # Load symbols
    symbols = load_symbols(args.symbols_file)
    print(f"\n=== Batch Download ===")
    print(f"Loaded {len(symbols)} symbols from {args.symbols_file}\n")

    if args.all or args.prices:
        print("=== Downloading Prices ===")
        results = batch_download_prices(
            symbols,
            skip_existing=args.skip_existing,
            period=args.period
        )
        print(f"\nPrices Results:")
        print(f"  Success: {results['success']}/{results['total']}")
        print(f"  Records: {results['total_records']:,}")
        print(f"  Failed: {results['failed']}")
        if results["errors"][:5]:
            print(f"  Sample errors: {results['errors'][:5]}")

    if args.all or args.fundamentals:
        print("\n=== Downloading Fundamentals ===")
        results = batch_download_fundamentals(symbols)
        print(f"\nFundamentals Results:")
        print(f"  Success: {results['success']}/{results['total']}")
        print(f"  Failed: {results['failed']}")

    # Show final stats
    db = get_db_manager()
    with db.get_session() as session:
        stats = db.get_statistics(session)
        print(f"\n=== Database Statistics ===")
        print(f"  Symbols: {stats['symbols']:,}")
        print(f"  Price Records: {stats['price_records']:,}")
        print(f"  Fundamentals: {stats['fundamentals']:,}")

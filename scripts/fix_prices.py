"""
Force update all prices with correct EOD data from Yahoo Finance.
With rate limiting protection.
"""
import yfinance as yf
from datetime import date, datetime, timedelta
from src.database import get_db_manager, Symbol, PriceHistory
import logging
import time
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

db = get_db_manager()

# Resume from symbol (optional argument)
start_from = sys.argv[1] if len(sys.argv) > 1 else None

# Get all symbols
with db.get_session() as session:
    symbols = session.query(Symbol).order_by(Symbol.code).all()
    symbol_list = [(s.id, s.code) for s in symbols]

# Skip to start_from if specified
if start_from:
    skip_idx = next((i for i, (_, code) in enumerate(symbol_list) if code >= start_from), 0)
    symbol_list = symbol_list[skip_idx:]
    logger.info(f"Resuming from {start_from} (index {skip_idx})")

total = len(symbol_list)
logger.info(f"Updating prices for {total} symbols...")

# Last 5 trading days to update
end_date = date.today()
start_date = end_date - timedelta(days=10)

updated = 0
errors = 0

with db.get_session() as session:
    for i, (symbol_id, code) in enumerate(symbol_list, 1):
        try:
            # Get correct prices from Yahoo
            ticker = yf.Ticker(code)
            hist = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat())

            if hist.empty:
                continue

            for idx, row in hist.iterrows():
                price_date = idx.date()

                # Find existing record
                existing = session.query(PriceHistory).filter(
                    PriceHistory.symbol_id == symbol_id,
                    PriceHistory.date == price_date
                ).first()

                if existing:
                    # Update with correct values
                    existing.open = row['Open']
                    existing.high = row['High']
                    existing.low = row['Low']
                    existing.close = row['Close']
                    existing.adjusted_close = row['Close']
                    existing.volume = row['Volume']
                else:
                    # Insert new
                    new_price = PriceHistory(
                        symbol_id=symbol_id,
                        date=price_date,
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        adjusted_close=row['Close'],
                        volume=row['Volume']
                    )
                    session.add(new_price)

            updated += 1

            # Rate limiting: pause every request
            time.sleep(0.3)  # 300ms between requests

            if i % 50 == 0:
                session.commit()
                logger.info(f"Progress: {i}/{total} ({100*i/total:.1f}%) - Updated: {updated}, Errors: {errors}")
                time.sleep(1)  # Extra pause every 50

            if i % 500 == 0:
                time.sleep(5)  # Longer pause every 500

        except Exception as e:
            errors += 1
            error_msg = str(e)[:80]
            if "Rate limited" in error_msg or "Too Many" in error_msg:
                logger.warning(f"Rate limited at {code}, waiting 60s...")
                time.sleep(60)
                # Retry
                try:
                    ticker = yf.Ticker(code)
                    hist = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat())
                    if not hist.empty:
                        for idx, row in hist.iterrows():
                            price_date = idx.date()
                            existing = session.query(PriceHistory).filter(
                                PriceHistory.symbol_id == symbol_id,
                                PriceHistory.date == price_date
                            ).first()
                            if existing:
                                existing.close = row['Close']
                                existing.open = row['Open']
                                existing.high = row['High']
                                existing.low = row['Low']
                        updated += 1
                        errors -= 1
                except:
                    pass
            elif errors <= 20:
                logger.warning(f"{code}: {error_msg}")

    session.commit()

logger.info(f"Done! Updated: {updated}, Errors: {errors}")

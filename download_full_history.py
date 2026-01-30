"""
Download full historical data for all symbols.
Runs with rate limiting protection.
"""
import yfinance as yf
from datetime import datetime
from src.database import get_db_manager, Symbol, PriceHistory
from sqlalchemy import func
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

db = get_db_manager()

# Get symbols that might need more history (less than 5000 records)
with db.get_session() as session:
    symbols_need_history = []

    all_symbols = session.query(Symbol).all()

    for s in all_symbols:
        count = session.query(func.count(PriceHistory.id)).filter(
            PriceHistory.symbol_id == s.id
        ).scalar()

        # If less than 5000 records, probably needs full history
        if count < 5000:
            symbols_need_history.append((s.id, s.code, count))

symbols_need_history.sort(key=lambda x: x[2])  # Sort by record count

total = len(symbols_need_history)
logger.info(f"Found {total} symbols that may need more historical data")

updated = 0
errors = 0

with db.get_session() as session:
    for i, (symbol_id, code, current_count) in enumerate(symbols_need_history, 1):
        try:
            ticker = yf.Ticker(code)
            hist = ticker.history(period='max')

            if hist.empty:
                continue

            count_new = 0
            for idx, row in hist.iterrows():
                price_date = idx.date()

                existing = session.query(PriceHistory).filter(
                    PriceHistory.symbol_id == symbol_id,
                    PriceHistory.date == price_date
                ).first()

                if not existing:
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
                    count_new += 1

            if count_new > 0:
                updated += 1

            time.sleep(0.5)

            if i % 50 == 0:
                session.commit()
                logger.info(f"Progress: {i}/{total} ({100*i/total:.1f}%) - Updated: {updated}")
                time.sleep(2)

            if i % 200 == 0:
                time.sleep(10)

        except Exception as e:
            errors += 1
            error_msg = str(e)
            if "Rate limited" in error_msg or "Too Many" in error_msg:
                logger.warning(f"Rate limited, waiting 120s...")
                session.commit()
                time.sleep(120)
            elif errors <= 20:
                logger.warning(f"{code}: {error_msg[:50]}")

    session.commit()

logger.info(f"Done! Updated: {updated}, Errors: {errors}")

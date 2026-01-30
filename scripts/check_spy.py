import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol
from datetime import date

db = get_db_manager()
with db.get_session() as session:
    spy = session.query(Symbol).filter(Symbol.code == 'SPY').first()
    if spy:
        prices = db.get_price_history(session, spy.id, start_date=date(2025, 12, 1))
        print(f'SPY prices count: {len(prices)}')
        print(f'Date range: {prices.index.min()} to {prices.index.max()}')

        today = date(2026, 1, 28)
        filtered = prices[
            (prices.index.date >= date(2025, 12, 31)) &
            (prices.index.date < today) &
            (prices.index.dayofweek < 5)
        ]
        print(f'Filtered count (weekdays from 31/12 to 27/01): {len(filtered)}')
        print('Dates:')
        for idx in filtered.index:
            print(f'  {idx.date()}')

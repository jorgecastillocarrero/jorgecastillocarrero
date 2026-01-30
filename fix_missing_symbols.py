"""Download missing symbols for initial value calculation"""
import sys
import os

os.chdir('C:/Users/usuario/financial-data-project')
sys.path.insert(0, 'C:/Users/usuario/financial-data-project')

from src.database import get_db_manager, Symbol
from src.eodhd_client import EODHDClient

# Missing symbols to download
MISSING_SYMBOLS = [
    'SHLG',   # Siemens Healthineers ADR
    'USAC',   # USA Compression Partners
]

db = get_db_manager()
client = EODHDClient()

with db.get_session() as session:
    for ticker in MISSING_SYMBOLS:
        print(f"\nDownloading {ticker}...")
        try:
            # Check if exists
            symbol = session.query(Symbol).filter(Symbol.code == ticker).first()
            if not symbol:
                symbol = Symbol(code=ticker, name=ticker, exchange='US')
                session.add(symbol)
                session.flush()
                print(f"  Created symbol {ticker}")

            # Download prices
            prices = client.get_eod_data(f"{ticker}.US", period='3mo')
            if prices is not None and not prices.empty:
                db.save_prices(session, symbol.id, prices)
                print(f"  Downloaded {len(prices)} prices")
            else:
                print(f"  No data found")
        except Exception as e:
            print(f"  Error: {e}")

    session.commit()
    print("\nDone!")

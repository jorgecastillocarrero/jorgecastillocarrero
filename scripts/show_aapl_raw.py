"""Show AAPL raw_data from fundamentals."""
import sys
sys.path.insert(0, r'C:\Users\usuario\financial-data-project')

from src.database import get_db_manager, Fundamental, Symbol
import json

db = get_db_manager()
with db.get_session() as session:
    sym = session.query(Symbol).filter(Symbol.code == 'AAPL').first()
    if sym:
        fund = session.query(Fundamental).filter(
            Fundamental.symbol_id == sym.id
        ).order_by(Fundamental.data_date.desc()).first()

        if fund:
            print(f"Date: {fund.data_date}")
            print(f"Dividend Yield in DB: {fund.dividend_yield}")
            print(f"\n=== RAW DATA ===\n")
            if fund.raw_data:
                data = json.loads(fund.raw_data)
                print(json.dumps(data, indent=2, default=str))
            else:
                print("No raw_data stored")
        else:
            print("No fundamentals found for AAPL")
    else:
        print("AAPL symbol not found")

"""
Import Interactive Brokers data from CSV statement.
Imports TLT as Cash equivalent and futures trades.
"""
import os
import sys
import csv
from datetime import datetime, date

os.chdir('C:/Users/usuario/financial-data-project')
sys.path.insert(0, 'C:/Users/usuario/financial-data-project')

from src.database import get_db_manager, IBAccount, IBTrade, IBFuturesTrade

def parse_ib_csv(filepath):
    """Parse IB statement CSV file."""
    trades = []
    futures_trades = []
    account_info = {}

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue

            section = row[0]
            row_type = row[1]

            # Account info
            if section == "Información sobre la cuenta" and row_type == "Data":
                if len(row) >= 4:
                    field = row[2]
                    value = row[3]
                    if field == "Cuenta":
                        account_info['account_id'] = value
                    elif field == "Nombre":
                        account_info['name'] = value
                    elif field == "Divisa base":
                        account_info['base_currency'] = value

            # Stock/ETF trades
            if section == "Operaciones" and row_type == "Data":
                if len(row) >= 15 and row[2] == "Order":
                    asset_type = row[3]
                    currency = row[4]
                    symbol = row[5]

                    if asset_type == "Acciones":
                        trade_datetime = row[6]
                        quantity = row[7].replace(",", "").replace('"', '')
                        price = float(row[8].replace(",", ""))
                        commission = float(row[11].replace(",", "")) if row[11] else 0

                        # Parse quantity (negative for sells)
                        qty = int(quantity) if quantity else 0

                        trades.append({
                            'symbol': symbol,
                            'trade_datetime': trade_datetime,
                            'quantity': qty,
                            'price': price,
                            'currency': currency,
                            'commission': abs(commission),
                            'asset_type': 'ETF' if symbol == 'TLT' else 'Stock',
                            'is_cash_equivalent': symbol == 'TLT'
                        })

                    elif asset_type == "Futuros":
                        trade_datetime = row[6]
                        quantity = row[7]
                        price = float(row[8].replace(",", ""))
                        commission = float(row[12].replace(",", "")) if row[12] else 0
                        code = row[15] if len(row) > 15 else ""

                        # Parse quantity
                        qty = int(quantity) if quantity else 0

                        # O = Open, C = Close
                        trade_type = "OPEN" if "O" in code else "CLOSE" if "C" in code else "UNKNOWN"

                        futures_trades.append({
                            'symbol': symbol,
                            'trade_datetime': trade_datetime,
                            'quantity': qty,
                            'price': price,
                            'currency': currency,
                            'commission': abs(commission),
                            'trade_type': trade_type,
                            'underlying': 'GC',  # Gold
                            'multiplier': 100
                        })

    return account_info, trades, futures_trades


def import_ib_data(filepath):
    """Import IB data into database."""
    print(f"Parsing {filepath}...")
    account_info, trades, futures_trades = parse_ib_csv(filepath)

    print(f"\nAccount: {account_info.get('account_id')} - {account_info.get('name')}")
    print(f"Trades found: {len(trades)}")
    print(f"Futures trades found: {len(futures_trades)}")

    db = get_db_manager()
    db.create_tables()  # Ensure new tables exist

    with db.get_session() as session:
        # Create or get account
        account = session.query(IBAccount).filter(
            IBAccount.account_id == account_info.get('account_id')
        ).first()

        if not account:
            account = IBAccount(
                account_id=account_info.get('account_id', 'UNKNOWN'),
                name=account_info.get('name', 'Unknown'),
                base_currency=account_info.get('base_currency', 'EUR')
            )
            session.add(account)
            session.flush()
            print(f"\nCreated new IB account: {account.account_id}")
        else:
            print(f"\nUsing existing IB account: {account.account_id}")

        # Import stock/ETF trades
        trades_added = 0
        for t in trades:
            try:
                trade_dt = datetime.strptime(t['trade_datetime'], "%Y-%m-%d, %H:%M:%S")
            except:
                continue

            # Check if already exists
            existing = session.query(IBTrade).filter(
                IBTrade.account_id == account.id,
                IBTrade.symbol == t['symbol'],
                IBTrade.trade_date == trade_dt,
                IBTrade.quantity == t['quantity']
            ).first()

            if not existing:
                trade = IBTrade(
                    account_id=account.id,
                    symbol=t['symbol'],
                    trade_date=trade_dt,
                    quantity=t['quantity'],
                    price=t['price'],
                    currency=t['currency'],
                    commission=t['commission'],
                    total_cost=abs(t['quantity'] * t['price']) + t['commission'],
                    trade_type='BUY' if t['quantity'] > 0 else 'SELL',
                    asset_type=t['asset_type'],
                    is_cash_equivalent=t['is_cash_equivalent']
                )
                session.add(trade)
                trades_added += 1

        print(f"Stock/ETF trades added: {trades_added}")

        # Import futures trades
        futures_added = 0
        for ft in futures_trades:
            try:
                trade_dt = datetime.strptime(ft['trade_datetime'], "%Y-%m-%d, %H:%M:%S")
            except:
                continue

            # Check if already exists
            existing = session.query(IBFuturesTrade).filter(
                IBFuturesTrade.account_id == account.id,
                IBFuturesTrade.symbol == ft['symbol'],
                IBFuturesTrade.trade_date == trade_dt,
                IBFuturesTrade.quantity == ft['quantity']
            ).first()

            if not existing:
                # Parse expiry from symbol (GCH6 -> March 2026)
                symbol = ft['symbol']
                expiry = None
                if len(symbol) >= 4:
                    month_code = symbol[2]  # H=Mar, J=Apr, etc.
                    year_code = symbol[3]   # 6=2026
                    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                                'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
                    if month_code in month_map:
                        exp_month = month_map[month_code]
                        exp_year = 2020 + int(year_code)
                        expiry = date(exp_year, exp_month, 28)  # Approximate

                futures_trade = IBFuturesTrade(
                    account_id=account.id,
                    symbol=ft['symbol'],
                    underlying=ft['underlying'],
                    expiry=expiry,
                    trade_date=trade_dt,
                    quantity=ft['quantity'],
                    price=ft['price'],
                    multiplier=ft['multiplier'],
                    currency=ft['currency'],
                    commission=ft['commission'],
                    notional_value=abs(ft['quantity'] * ft['price'] * ft['multiplier']),
                    trade_type=ft['trade_type'],
                    is_day_trade=True
                )
                session.add(futures_trade)
                futures_added += 1

        print(f"Futures trades added: {futures_added}")

        session.commit()
        print("\nImport completed successfully!")

        # Summary
        print("\n" + "="*60)
        print("RESUMEN DE IMPORTACION")
        print("="*60)

        # TLT position
        tlt_trades = session.query(IBTrade).filter(
            IBTrade.account_id == account.id,
            IBTrade.symbol == 'TLT'
        ).all()

        total_tlt_shares = sum(t.quantity for t in tlt_trades)
        total_tlt_cost = sum(t.total_cost for t in tlt_trades if t.quantity > 0)
        avg_price = total_tlt_cost / total_tlt_shares if total_tlt_shares > 0 else 0

        print(f"\nTLT (Cash equivalent):")
        print(f"  Shares: {total_tlt_shares:,}")
        print(f"  Total cost: ${total_tlt_cost:,.2f}")
        print(f"  Avg price: ${avg_price:.2f}")

        # Futures summary
        futures = session.query(IBFuturesTrade).filter(
            IBFuturesTrade.account_id == account.id
        ).all()

        # Group by symbol
        by_symbol = {}
        for ft in futures:
            if ft.symbol not in by_symbol:
                by_symbol[ft.symbol] = {'open': 0, 'close': 0, 'trades': 0}
            by_symbol[ft.symbol]['trades'] += 1
            if ft.trade_type == 'OPEN':
                by_symbol[ft.symbol]['open'] += 1
            else:
                by_symbol[ft.symbol]['close'] += 1

        print(f"\nFuturos (Day trades):")
        for symbol, data in by_symbol.items():
            print(f"  {symbol}: {data['trades']} trades ({data['open']} opens, {data['close']} closes)")


if __name__ == "__main__":
    filepath = r"C:\Users\usuario\Downloads\U17236599_20260101_20260126.csv"
    import_ib_data(filepath)

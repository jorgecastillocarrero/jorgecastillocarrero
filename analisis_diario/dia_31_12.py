import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_db_manager, Symbol, AccountHolding, AccountCash, PortfolioSnapshot
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date

CASH_EN_TRANSITO = 190000

db = get_db_manager()
portfolio_service = get_portfolio_service(db)
td = date(2025, 12, 31)

with db.get_session() as session:
    eur_usd = portfolio_service.get_symbol_price('EURUSD=X', td) or 1.035

    print(f"DIA: {td.strftime('%d/%m/%Y')}")
    print(f"EUR/USD: {eur_usd:.4f}")
    print("="*70)

    holdings = session.query(AccountHolding).all()
    cash_entries = session.query(AccountCash).all()

    type_values = {}
    type_detail = {}  # {type: [(symbol, shares, price, value_eur)]}

    for holding in holdings:
        symbol = holding.symbol
        shares = holding.shares
        asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')

        price = portfolio_service.get_symbol_price(symbol, td)
        if price is None:
            print(f"  SIN PRECIO: {symbol}")
            continue

        value = price * shares
        currency = 'EUR'

        symbol_obj = session.query(Symbol).filter(Symbol.code == symbol).first()
        if symbol_obj:
            currency = symbol_obj.currency or 'EUR'

        value_eur = value
        if currency == 'USD':
            value_eur = value / eur_usd

        if asset_type not in type_values:
            type_values[asset_type] = 0.0
            type_detail[asset_type] = []
        type_values[asset_type] += value_eur
        type_detail[asset_type].append((symbol, shares, price, value_eur, currency))

    # Cash
    cash_total = 0
    print("\nCASH:")
    for c in cash_entries:
        if c.currency == 'EUR':
            cash_total += c.amount
            print(f"  {c.account_code} EUR: {c.amount:,.2f}")
        elif c.currency == 'USD':
            val_eur = c.amount / eur_usd
            cash_total += val_eur
            print(f"  {c.account_code} USD: {c.amount:,.2f} = {val_eur:,.2f} EUR")

    if 'Cash/Monetario' not in type_values:
        type_values['Cash/Monetario'] = 0.0
    type_values['Cash/Monetario'] += cash_total

    print(f"  Cash en tránsito: {CASH_EN_TRANSITO:,.0f} EUR")
    type_values['Cash/Monetario'] += CASH_EN_TRANSITO

    # Resumen por tipo
    print(f"\nRESUMEN POR TIPO DE ACTIVO:")
    print(f"{'Tipo':<18} {'Valor EUR':>15}")
    print("-"*35)
    total = 0
    for t in ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario']:
        val = type_values.get(t, 0)
        total += val
        print(f"{t:<18} {val:>15,.0f}")
    print("-"*35)
    print(f"{'TOTAL CALCULO':<18} {total:>15,.0f}")

    # Snapshot
    snap = session.query(PortfolioSnapshot).filter(PortfolioSnapshot.snapshot_date == td).all()
    snapshot_total = sum(s.total_value for s in snap)
    print(f"{'SNAPSHOT':<18} {snapshot_total:>15,.0f}")
    print(f"{'DIFERENCIA':<18} {total - snapshot_total:>+15,.0f}")

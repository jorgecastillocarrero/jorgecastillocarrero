import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol, AccountHolding, AccountCash
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

date1 = date(2026, 1, 5)
date2 = date(2026, 1, 6)

print(f"VARIACION POR TIPO DE ACTIVO: {date1.strftime('%d/%m/%Y')} vs {date2.strftime('%d/%m/%Y')}")
print("="*90)

with db.get_session() as session:
    # Get EUR/USD rates
    eur_usd_1 = portfolio_service.get_symbol_price('EURUSD=X', date1) or 1.0350
    eur_usd_2 = portfolio_service.get_symbol_price('EURUSD=X', date2) or 1.0350

    print(f"EUR/USD {date1.strftime('%d/%m')}: {eur_usd_1:.4f}")
    print(f"EUR/USD {date2.strftime('%d/%m')}: {eur_usd_2:.4f}")
    print()

    # Get all holdings
    holdings = session.query(AccountHolding).all()

    type_values_1 = {}
    type_values_2 = {}

    for holding in holdings:
        symbol = holding.symbol
        shares = holding.shares
        asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')

        price_1 = portfolio_service.get_symbol_price(symbol, date1)
        price_2 = portfolio_service.get_symbol_price(symbol, date2)

        if price_1 is None or price_2 is None:
            continue

        value_1 = price_1 * shares
        value_2 = price_2 * shares

        # Currency conversion
        symbol_obj = session.query(Symbol).filter(Symbol.code == symbol).first()
        if symbol_obj and symbol_obj.currency == 'USD':
            value_1 = value_1 / eur_usd_1
            value_2 = value_2 / eur_usd_2

        if asset_type not in type_values_1:
            type_values_1[asset_type] = 0
            type_values_2[asset_type] = 0

        type_values_1[asset_type] += value_1
        type_values_2[asset_type] += value_2

    # Add Cash
    cash_entries = session.query(AccountCash).all()
    cash_eur_1 = 0
    cash_eur_2 = 0
    for c in cash_entries:
        if c.currency == 'EUR':
            cash_eur_1 += c.amount
            cash_eur_2 += c.amount
        elif c.currency == 'USD':
            cash_eur_1 += c.amount / eur_usd_1
            cash_eur_2 += c.amount / eur_usd_2

    # Add cash to Cash/Monetario
    if 'Cash/Monetario' not in type_values_1:
        type_values_1['Cash/Monetario'] = 0
        type_values_2['Cash/Monetario'] = 0
    type_values_1['Cash/Monetario'] += cash_eur_1
    type_values_2['Cash/Monetario'] += cash_eur_2

    # Añadir cuenta en tránsito para cuadrar con snapshots
    # Calcular diferencia con snapshot para cada fecha
    from src.database import PortfolioSnapshot
    snap1 = session.query(PortfolioSnapshot).filter(PortfolioSnapshot.snapshot_date == date1).all()
    snap2 = session.query(PortfolioSnapshot).filter(PortfolioSnapshot.snapshot_date == date2).all()
    snapshot_total_1 = sum(s.total_value for s in snap1)
    snapshot_total_2 = sum(s.total_value for s in snap2)

    calc_total_1 = sum(type_values_1.values())
    calc_total_2 = sum(type_values_2.values())

    ajuste_1 = snapshot_total_1 - calc_total_1
    ajuste_2 = snapshot_total_2 - calc_total_2

    type_values_1['Cash/Monetario'] += ajuste_1
    type_values_2['Cash/Monetario'] += ajuste_2

# Print results
print(f"{'Tipo':<20} {'Valor '+date1.strftime('%d/%m'):>15} {'Valor '+date2.strftime('%d/%m'):>15} {'Var. EUR':>12} {'Var. %':>10}")
print("-"*90)

total_1 = 0
total_2 = 0

for asset_type in ['Mensual', 'Quant', 'Value', 'Alpha Picks', 'Oro/Mineras', 'Cash/Monetario']:
    val_1 = type_values_1.get(asset_type, 0)
    val_2 = type_values_2.get(asset_type, 0)
    var_eur = val_2 - val_1
    var_pct = ((val_2 / val_1) - 1) * 100 if val_1 > 0 else 0

    total_1 += val_1
    total_2 += val_2

    print(f"{asset_type:<20} {val_1:>15,.0f} {val_2:>15,.0f} {var_eur:>+12,.0f} {var_pct:>+10.2f}%")

print("-"*90)
total_var_eur = total_2 - total_1
total_var_pct = ((total_2 / total_1) - 1) * 100 if total_1 > 0 else 0
print(f"{'TOTAL':<20} {total_1:>15,.0f} {total_2:>15,.0f} {total_var_eur:>+12,.0f} {total_var_pct:>+10.2f}%")

print()
print(f"Snapshot {date1.strftime('%d/%m')}: {snapshot_total_1:,.0f} EUR")
print(f"Snapshot {date2.strftime('%d/%m')}: {snapshot_total_2:,.0f} EUR")

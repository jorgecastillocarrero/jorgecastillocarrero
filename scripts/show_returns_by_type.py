import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_manager, Symbol, AccountHolding
from src.portfolio_data import get_portfolio_service, ASSET_TYPE_MAP
from datetime import date
import pandas as pd

db = get_db_manager()
portfolio_service = get_portfolio_service(db)

target_date = date(2026, 1, 2)
initial_date = date(2025, 12, 31)

print(f"RENTABILIDAD POR TIPO DE ACTIVO - {target_date.strftime('%d/%m/%Y')}")
print("="*80)

# Get all holdings from database
with db.get_session() as session:
    holdings = session.query(AccountHolding).all()

    # Group by asset type
    type_values_initial = {}
    type_values_current = {}

    for holding in holdings:
        symbol = holding.symbol
        shares = holding.shares
        asset_type = ASSET_TYPE_MAP.get(symbol, 'Otros')

        # Get prices
        price_initial = portfolio_service.get_symbol_price(symbol, initial_date)
        price_current = portfolio_service.get_symbol_price(symbol, target_date)

        if price_initial and price_current:
            value_initial = price_initial * shares
            value_current = price_current * shares

            # Handle currency conversion for USD symbols
            symbol_obj = session.query(Symbol).filter(Symbol.code == symbol).first()
            if symbol_obj and symbol_obj.currency == 'USD':
                # Get EUR/USD rates
                eur_usd_initial = portfolio_service.get_symbol_price('EURUSD=X', initial_date) or 1.035
                eur_usd_current = portfolio_service.get_symbol_price('EURUSD=X', target_date) or 1.035
                value_initial = value_initial / eur_usd_initial
                value_current = value_current / eur_usd_current

            if asset_type not in type_values_initial:
                type_values_initial[asset_type] = 0
                type_values_current[asset_type] = 0

            type_values_initial[asset_type] += value_initial
            type_values_current[asset_type] += value_current

# Calculate returns by type
print(f"\n{'Tipo':<20} {'Valor 31/12':>15} {'Valor 02/01':>15} {'Rent. EUR':>12} {'Rent. %':>10}")
print("-"*80)

total_initial = 0
total_current = 0

for asset_type in sorted(type_values_initial.keys()):
    val_ini = type_values_initial[asset_type]
    val_cur = type_values_current[asset_type]
    rent_eur = val_cur - val_ini
    rent_pct = ((val_cur / val_ini) - 1) * 100 if val_ini > 0 else 0

    total_initial += val_ini
    total_current += val_cur

    print(f"{asset_type:<20} {val_ini:>15,.0f} {val_cur:>15,.0f} {rent_eur:>+12,.0f} {rent_pct:>+10.2f}%")

print("-"*80)
total_rent_eur = total_current - total_initial
total_rent_pct = ((total_current / total_initial) - 1) * 100 if total_initial > 0 else 0
print(f"{'TOTAL':<20} {total_initial:>15,.0f} {total_current:>15,.0f} {total_rent_eur:>+12,.0f} {total_rent_pct:>+10.2f}%")
